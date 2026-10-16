// Compare tab: re-render samples through the user's Forge once per epoch checkpoint, loss charts from
// TensorBoard logs, dataset snapshots, and finding datasets that moved since a job was trained.
//
// Per job, everything lives in jobs/<job>/compare/:
//   compare.json            { template, weight, auto, samples: [{ id, kind, prompt, negative, infotext?, width?, height?, original }] }
//   originals/<id>.<ext>    the image each sample is compared against
//   renders/<id>/<ckpt>.png one render per checkpoint ("__base__" = no LoRA)
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const TOML = require('@iarna/toml');
const { readScalars } = require('./tbevents');
const { datasetInfo } = require('./datasetinfo');

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;
const BASE = '__base__';

module.exports = function registerCompare(app, { getJobPath, getGlobalConfig, sanitizeName, runningJobs, getPython, searchRoot }) {
    // ---------- Forge ----------
    function forgeUrl() {
        return ((getGlobalConfig().forge || {}).url || 'http://127.0.0.1:7860').replace(/\/+$/, '');
    }

    async function forge(endpoint, body) {
        const r = await fetch(forgeUrl() + endpoint, body === undefined ? {} : {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(`Forge ${endpoint} -> ${r.status}: ${(await r.text()).slice(0, 300)}`);
        return r.json();
    }

    // Forge's Lora folder, worked out from the LoRAs it already lists (Forge must be on this PC). [forge] lora_dir overrides.
    async function forgeLoraDir() {
        const own = (getGlobalConfig().forge || {}).lora_dir;
        if (own) return own;
        const loras = await forge('/sdapi/v1/loras');
        if (!loras.length) throw new Error('Forge lists no LoRAs, so its Lora folder is unknown. Set lora_dir under [forge] in global_config.toml');
        const dir = loras.map(l => path.dirname(l.path)).sort((a, b) => a.length - b.length)[0];
        if (!fs.existsSync(dir)) throw new Error(`Forge's Lora folder ${dir} isn't on this PC`);
        return dir;
    }

    async function pngInfo(dataUrl) {
        const info = await forge('/sdapi/v1/png-info', { image: dataUrl });
        if (!info.info) throw new Error('No generation info in this image (needs a PNG saved by Forge/A1111 with metadata)');
        return info;
    }

    // ---------- per-job storage ----------
    const jobDir = (job) => getJobPath(job);
    const cmpDir = (job) => path.join(jobDir(job), 'compare');
    const outDir = (job) => path.join(jobDir(job), 'output');

    function load(job) {
        const p = path.join(cmpDir(job), 'compare.json');
        const cfg = fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : {};
        return { template: null, weight: 1, auto: true, samples: [], ...cfg };
    }
    function save(job, cfg) {
        fs.mkdirSync(cmpDir(job), { recursive: true });
        fs.writeFileSync(path.join(cmpDir(job), 'compare.json'), JSON.stringify(cfg, null, 2));
    }
    const clearRenders = (job, id) => fs.rmSync(path.join(cmpDir(job), 'renders', ...(id ? [id] : [])), { recursive: true, force: true });

    function saveDataUrl(file, dataUrl) {
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, Buffer.from(dataUrl.split(',')[1], 'base64'));
    }

    // Checkpoints in output/, oldest epoch first; the final save has no -NNNNNN suffix.
    // Files touched in the last 15s are skipped: they may still be being written.
    function checkpoints(job) {
        const dir = outDir(job);
        if (!fs.existsSync(dir)) return [];
        const now = Date.now();
        return fs.readdirSync(dir)
            .filter(f => f.endsWith('.safetensors'))
            .map(f => {
                const mtime = fs.statSync(path.join(dir, f)).mtimeMs;
                const name = f.slice(0, -'.safetensors'.length);
                const m = name.match(/-(\d{6})$/);
                return { name, file: path.join(dir, f), mtime, epoch: m ? +m[1] : null };
            })
            .filter(c => now - c.mtime > 15000)
            .sort((a, b) => (a.epoch ?? Infinity) - (b.epoch ?? Infinity));
    }

    // Hardlinks checkpoints into Forge's Lora folder (_trainer/<job>/) so <lora:name:w> resolves.
    // Only for the duration of a render pass (unlinkCheckpoints after), so Forge's LoRA list stays yours.
    async function linkCheckpoints(job, ckpts) {
        const dst = path.join(await forgeLoraDir(), '_trainer', sanitizeName(job));
        fs.mkdirSync(dst, { recursive: true });
        let changed = false;
        for (const c of ckpts) {
            const to = path.join(dst, c.name + '.safetensors');
            if (fs.existsSync(to) && fs.statSync(to).ino === fs.statSync(c.file).ino) continue;
            fs.rmSync(to, { force: true });
            try { fs.linkSync(c.file, to); } catch { fs.copyFileSync(c.file, to); } // other drive -> copy
            changed = true;
        }
        if (changed) await forge('/sdapi/v1/refresh-loras', {});
    }

    async function unlinkCheckpoints() {
        const dir = path.join(await forgeLoraDir(), '_trainer');
        if (!fs.existsSync(dir)) return;
        fs.rmSync(dir, { recursive: true, force: true }); // hardlinks: the job's own files are untouched
        await forge('/sdapi/v1/refresh-loras', {});
    }

    function renderPath(job, id, ckpt) {
        return path.join(cmpDir(job), 'renders', id, ckpt + '.png');
    }

    // ---------- render queue (one Forge, so one render at a time across all jobs) ----------
    const pending = new Map(); // job -> { report }
    const state = {}; // job -> { running, done, total, label, error, stop }
    let busy = false;

    function kick(job, report = false) {
        pending.set(job, { report: report || pending.get(job)?.report || false });
        pump();
    }

    async function pump() {
        if (busy) return;
        busy = true;
        try {
            while (pending.size) {
                const [job, opts] = pending.entries().next().value;
                pending.delete(job);
                const st = state[job] = { running: true, done: 0, total: 0, label: '', error: null, stop: false };
                try {
                    await renderMissing(job, st);
                } catch (e) {
                    st.error = e.message;
                }
                // the chart part of the report doesn't need Forge, so build it even if rendering failed
                if (opts.report && !st.stop) {
                    try { await makeReport(job, st); } catch (e) { st.error = `Report: ${e.message}`; }
                }
                st.running = false;
                st.label = '';
            }
        } finally {
            busy = false;
        }
    }

    async function renderMissing(job, st) {
        const cfg = load(job);
        if (!cfg.samples.length) return;
        const ckpts = checkpoints(job);
        const todo = [];
        for (const s of cfg.samples) {
            for (const c of [{ name: BASE, mtime: 0 }, ...ckpts]) {
                const out = renderPath(job, s.id, c.name);
                if (!fs.existsSync(out) || fs.statSync(out).mtimeMs < c.mtime) todo.push({ s, c, out });
            }
        }
        st.total = todo.length;
        if (!todo.length) return;
        await linkCheckpoints(job, ckpts);
        try {
            await renderAll(job, cfg, todo, st);
        } finally {
            await unlinkCheckpoints().catch(e => console.error('[compare] unlink:', e.message));
        }
    }

    async function renderAll(job, cfg, todo, st) {
        for (const { s, c, out } of todo) {
            if (st.stop) return;
            st.label = `${c.name === BASE ? 'No LoRA' : c.name} · sample ${cfg.samples.indexOf(s) + 1}`;
            const infotext = s.infotext || cfg.template?.infotext;
            if (!infotext) throw new Error('Dataset samples need a settings template: drop a Forge image on "Settings template" first');
            const body = {
                infotext,
                prompt: c.name === BASE ? s.prompt : `${s.prompt} <lora:${c.name}:${cfg.weight}>`,
                negative_prompt: s.negative ?? cfg.template?.negative ?? '',
                batch_size: 1, n_iter: 1, save_images: false, send_images: true,
                // Forge Neo's infotext parser crashes on list-typed fields, so set those ourselves
                styles: [], hr_additional_modules: ['Use same choices'],
            };
            if (s.width) Object.assign(body, { width: s.width, height: s.height });
            const r = await forge('/sdapi/v1/txt2img', body);
            fs.mkdirSync(path.dirname(out), { recursive: true });
            fs.writeFileSync(out, Buffer.from(r.images[0], 'base64'));
            st.done++;
        }
    }

    // New checkpoints while training -> render them (if the job opted in)
    setInterval(() => {
        for (const [job, info] of runningJobs) {
            if (info.type !== 'training') continue;
            const cfg = load(job);
            if (cfg.auto && cfg.samples.length && !state[job]?.running) kick(job);
        }
    }, 30000);

    // ---------- losses ----------
    // -> { current: [[step, v]], epochs: [{ epoch, step, train, check }] }  ("check" = fixed-noise/validation loss)
    function losses(job) {
        const s = readScalars(path.join(jobDir(job), 'logs'));
        const cur = s['loss/current'] || [];
        // epoch events are logged with step = epoch; place them on the step axis by wall time
        const stepAt = (wall) => {
            let best = 0;
            for (const p of cur) if (p.wall <= wall + 1) best = p.step;
            return best;
        };
        const check = new Map((s['loss/validation/epoch_average'] || []).map(p => [p.step, p.value]));
        return {
            current: cur.map(p => [p.step, p.value]),
            epochs: (s['loss/epoch_average'] || []).map(p => ({ epoch: p.step, step: stepAt(p.wall), train: p.value, check: check.get(p.step) ?? null })),
        };
    }

    // ---------- report (chart + epoch sheet PNG next to the LoRAs) ----------
    function runPython(args) {
        return new Promise((resolve, reject) => {
            const p = spawn(getPython(), args, { cwd: path.join(__dirname, '..') });
            let err = '';
            p.stderr.on('data', d => { err += d; });
            p.on('close', code => (code === 0 ? resolve() : reject(new Error(err.slice(-500) || `python exited ${code}`))));
            p.on('error', reject);
        });
    }

    async function makeReport(job, st) {
        const cfg = load(job);
        const ckpts = checkpoints(job);
        if (!ckpts.length) return;
        st.label = 'Building report';
        const outName = ckpts[ckpts.length - 1].name.replace(/-\d{6}$/, '');
        const data = path.join(cmpDir(job), 'report.json');
        fs.writeFileSync(data, JSON.stringify({
            title: outName,
            losses: losses(job),
            columns: [BASE, ...ckpts.map(c => c.name)],
            samples: cfg.samples.map(s => ({
                original: path.join(cmpDir(job), s.original),
                renders: [BASE, ...ckpts.map(c => c.name)].map(n => renderPath(job, s.id, n)),
            })),
        }));
        await runPython([path.join('tools', 'report.py'), data, path.join(outDir(job), `${outName}_report.png`)]);
    }

    // ---------- dataset ----------
    function subsets(job) {
        const p = path.join(jobDir(job), 'dataset.toml');
        if (!fs.existsSync(p)) return [];
        const ds = TOML.parse(fs.readFileSync(p, 'utf8'));
        const out = [];
        for (const d of [].concat(ds.datasets || [])) {
            for (const s of [].concat(d.subsets || [])) out.push({ dir: s.image_dir, ext: d.caption_extension || '.txt' });
        }
        return out;
    }

    function readCaption(dir, file, ext) {
        const p = path.join(dir, file.replace(/\.[^.]+$/, ext));
        return fs.existsSync(p) ? fs.readFileSync(p, 'utf8').trim() : '';
    }

    // The 12 numbered category folders ("1. GENDER" ... "12. SYMBOLS") that get compiled into one training folder
    function categoryDirs(base) {
        if (!fs.existsSync(base)) return [];
        return fs.readdirSync(base, { withFileTypes: true })
            .filter(e => e.isDirectory() && /^\d+\.\s/.test(e.name))
            .map(e => ({ n: parseInt(e.name), name: e.name, dir: path.join(base, e.name) }))
            .sort((a, b) => a.n - b.n);
    }

    // _tools/compiler.py, in place: images from images/ (or, in older datasets, any category folder),
    // captions = trigger.txt + each category's tags in folder order
    function compileCategories(base) {
        const merged = {};
        const images = {};
        const imgDir = path.join(base, 'images');
        if (fs.existsSync(imgDir)) for (const f of fs.readdirSync(imgDir)) if (IMAGE_EXT.test(f)) images[f] = path.join(imgDir, f);
        for (const c of categoryDirs(base)) {
            for (const f of fs.readdirSync(c.dir)) {
                if (IMAGE_EXT.test(f)) images[f] ||= path.join(c.dir, f);
                else if (f.endsWith('.txt')) {
                    const tags = fs.readFileSync(path.join(c.dir, f), 'utf8').replace(/\n/g, ',').split(',').map(t => t.trim()).filter(Boolean);
                    if (tags.length) (merged[f] ||= []).push(tags.join(', '));
                }
            }
        }
        // trigger.txt (e.g. "@jennieart") goes first in every caption, like _tools/compiler.py
        const tf = path.join(base, 'trigger.txt');
        const trigger = fs.existsSync(tf) ? fs.readFileSync(tf, 'utf8').trim() : '';
        const caption = (parts) => (trigger && !parts[0].startsWith(trigger) ? [trigger, ...parts] : parts).join(', ');
        return { images, captions: Object.fromEntries(Object.entries(merged).map(([f, parts]) => [f, caption(parts)])) };
    }

    // Captions (compiled + per category) and 256px thumbnails, copied at training start so a job
    // always shows what it was trained on.
    function snapshot(job) {
        const dst = path.join(jobDir(job), 'snapshots', new Date().toISOString().replace(/[:.]/g, '-'));
        subsets(job).forEach((s, i) => {
            if (!fs.existsSync(s.dir)) return;
            const sub = path.join(dst, `${i}-${path.basename(s.dir)}`);
            fs.mkdirSync(sub, { recursive: true });
            for (const f of fs.readdirSync(s.dir)) if (f.endsWith(s.ext)) fs.copyFileSync(path.join(s.dir, f), path.join(sub, f));
            for (const base of [path.dirname(s.dir), s.dir]) {
                for (const c of categoryDirs(base)) {
                    const cd = path.join(sub, 'categories', c.name);
                    fs.mkdirSync(cd, { recursive: true });
                    for (const f of fs.readdirSync(c.dir)) if (f.endsWith('.txt')) fs.copyFileSync(path.join(c.dir, f), path.join(cd, f));
                }
            }
            runPython([path.join('tools', 'report.py'), '--thumbs', s.dir, path.join(sub, 'thumbs')]).catch(e => console.error('[snapshot] thumbs:', e.message));
        });
    }

    // What the job's newest LoRA says it was trained on: { dirName: { count, tags: {tag: n} } }
    function trainedOn(job) {
        const files = [...checkpoints(job)].reverse();
        for (const c of files) {
            const fd = fs.openSync(c.file, 'r');
            try {
                const lenBuf = Buffer.alloc(8);
                fs.readSync(fd, lenBuf, 0, 8, 0);
                const len = Number(lenBuf.readBigUInt64LE(0));
                const head = Buffer.alloc(len);
                fs.readSync(fd, head, 0, len, 8);
                const meta = JSON.parse(head.toString('utf8')).__metadata__ || {};
                if (!meta.ss_tag_frequency) continue;
                const dirs = JSON.parse(meta.ss_dataset_dirs || '{}');
                const tags = JSON.parse(meta.ss_tag_frequency);
                return Object.fromEntries(Object.keys(tags).map(k => [k, { count: dirs[k]?.img_count ?? null, tags: tags[k] }]));
            } catch { /* not a readable safetensors header */ } finally { fs.closeSync(fd); }
        }
        return {};
    }

    function tagFreq(captions) {
        const f = {};
        for (const c of captions) for (const t of c.split(',').map(x => x.trim()).filter(Boolean)) f[t] = (f[t] || 0) + 1;
        return f;
    }
    function cosine(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (const [k, v] of Object.entries(a)) { na += v * v; if (b[k]) dot += v * b[k]; }
        for (const v of Object.values(b)) nb += v * v;
        return na && nb ? dot / Math.sqrt(na * nb) : 0;
    }

    // Folders under searchRoot that hold a dataset: compiled (images + captions) or uncompiled (numbered category folders)
    function datasetCandidates() {
        const out = [];
        const walk = (d, depth) => {
            let entries;
            try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
            if (categoryDirs(d).length >= 3) {
                const { images, captions } = compileCategories(d);
                out.push({ dir: d, kind: 'uncompiled', count: Object.keys(images).length, tags: tagFreq(Object.values(captions)) });
            }
            const imgs = entries.filter(e => e.isFile() && IMAGE_EXT.test(e.name));
            if (imgs.length && entries.some(e => e.name.endsWith('.txt'))) {
                out.push({ dir: d, kind: 'compiled', count: imgs.length, tags: tagFreq(imgs.map(e => readCaption(d, e.name, '.txt'))) });
            }
            if (depth >= 4) return;
            for (const e of entries) {
                if (e.isDirectory() && !/^(venv|node_modules|\.git|jobs|output|__pycache__|snapshots|compare|images|COMPILED)$/.test(e.name) && !/^\d+\.\s/.test(e.name)) walk(path.join(d, e.name), depth + 1);
            }
        };
        walk(searchRoot, 0);
        return out;
    }

    // ---------- routes ----------
    const route = (fn) => async (req, res) => {
        try { res.json(await fn(sanitizeName(req.params.name), req.body || {}, req)); } catch (e) { res.status(500).json({ error: e.message }); }
    };

    app.get('/api/jobs/:name/compare', route((job) => {
        const cfg = load(job);
        const ckpts = checkpoints(job);
        const renders = {};
        for (const s of cfg.samples) {
            renders[s.id] = [BASE, ...ckpts.map(c => c.name)].filter(n => fs.existsSync(renderPath(job, s.id, n)));
        }
        const st = state[job] || {};
        return {
            ...cfg,
            forgeUrl: forgeUrl(),
            training: runningJobs.get(job)?.type === 'training',
            checkpoints: ckpts.map(({ name, epoch }) => ({ name, epoch })),
            renders,
            state: { running: !!st.running || pending.has(job), done: st.done || 0, total: st.total || 0, label: st.label || '', error: st.error || null },
        };
    }));

    app.get('/api/jobs/:name/losses', route((job) => losses(job)));

    app.post('/api/jobs/:name/compare/template', route(async (job, body) => {
        const info = await pngInfo(body.image);
        const cfg = load(job);
        saveDataUrl(path.join(cmpDir(job), 'template.png'), body.image);
        cfg.template = {
            infotext: info.info,
            negative: info.parameters['Negative prompt'] || '',
            width: +info.parameters['Size-1'] || 1024,
            height: +info.parameters['Size-2'] || 1024,
            hrScale: +info.parameters['Hires upscale'] || 1, // dataset samples divide by this so the final size matches the original
            original: 'template.png',
        };
        save(job, cfg);
        clearRenders(job);
        return cfg.template;
    }));

    app.post('/api/jobs/:name/compare/samples', route(async (job, body) => {
        const cfg = load(job);
        const id = crypto.randomBytes(4).toString('hex');
        let sample;
        if (body.kind === 'forge') {
            const info = await pngInfo(body.image);
            sample = { id, kind: 'forge', prompt: info.parameters.Prompt || '', negative: info.parameters['Negative prompt'] || '', infotext: info.info, original: `originals/${id}.png` };
            saveDataUrl(path.join(cmpDir(job), sample.original), body.image);
        } else {
            const sub = subsets(job)[body.subset];
            const file = path.basename(body.file);
            if (!sub || !fs.existsSync(path.join(sub.dir, file))) throw new Error('Dataset image not found');
            sample = { id, kind: 'dataset', prompt: readCaption(sub.dir, file, sub.ext), negative: null, width: body.width, height: body.height, original: `originals/${id}${path.extname(file)}`, source: path.join(sub.dir, file) };
            fs.mkdirSync(path.join(cmpDir(job), 'originals'), { recursive: true });
            fs.copyFileSync(sample.source, path.join(cmpDir(job), sample.original));
        }
        cfg.samples.push(sample);
        save(job, cfg);
        return sample;
    }));

    app.put('/api/jobs/:name/compare/samples/:id', route((job, body, req) => {
        const cfg = load(job);
        const s = cfg.samples.find(x => x.id === req.params.id);
        if (!s) throw new Error('No such sample');
        if (typeof body.prompt === 'string') s.prompt = body.prompt;
        if ('negative' in body) s.negative = body.negative;
        save(job, cfg);
        clearRenders(job, s.id);
        return s;
    }));

    app.delete('/api/jobs/:name/compare/samples/:id', route((job, body, req) => {
        const cfg = load(job);
        const s = cfg.samples.find(x => x.id === req.params.id);
        cfg.samples = cfg.samples.filter(x => x.id !== req.params.id);
        save(job, cfg);
        clearRenders(job, req.params.id);
        if (s) fs.rmSync(path.join(cmpDir(job), s.original), { force: true });
        return { ok: true };
    }));

    app.put('/api/jobs/:name/compare/settings', route((job, body) => {
        const cfg = load(job);
        if (body.weight !== undefined && +body.weight !== cfg.weight) { cfg.weight = +body.weight; clearRenders(job); }
        if (body.auto !== undefined) cfg.auto = !!body.auto;
        save(job, cfg);
        return cfg;
    }));

    app.post('/api/jobs/:name/compare/render', route((job) => { kick(job); return { ok: true }; }));
    app.post('/api/jobs/:name/compare/report', route((job) => { kick(job, true); return { ok: true }; }));
    app.post('/api/jobs/:name/compare/stop', route((job) => {
        pending.delete(job);
        if (state[job]) state[job].stop = true;
        return { ok: true };
    }));

    // compare/<relative file> (originals, renders, template)
    app.get('/api/jobs/:name/compare/file', (req, res) => {
        const root = cmpDir(sanitizeName(req.params.name));
        const p = path.resolve(root, String(req.query.p || ''));
        if (!p.startsWith(root + path.sep) || !fs.existsSync(p)) return res.status(404).end();
        res.set('Cache-Control', 'no-cache').sendFile(p);
    });

    // Facts + suggested resolution/buckets for the Dataset tab (takes the folders as currently typed, saved or not)
    app.post('/api/dataset-info', (req, res) => {
        try { res.json(datasetInfo(req.body.subsets || [], +req.body.resolution || null)); } catch (e) { res.status(500).json({ error: e.message }); }
    });

    app.get('/api/jobs/:name/dataset-images', route((job) => subsets(job).map((s, i) => ({
        subset: i,
        dir: s.dir,
        exists: fs.existsSync(s.dir),
        files: fs.existsSync(s.dir) ? fs.readdirSync(s.dir).filter(f => IMAGE_EXT.test(f)).map(f => ({ file: f, caption: readCaption(s.dir, f, s.ext) })) : [],
    }))));

    app.get('/api/jobs/:name/dataset-image', (req, res) => {
        const sub = subsets(sanitizeName(req.params.name))[+req.query.s];
        const p = sub && path.join(sub.dir, path.basename(String(req.query.f || '')));
        if (!p || !IMAGE_EXT.test(p) || !fs.existsSync(p)) return res.status(404).end();
        res.set('Cache-Control', 'max-age=3600').sendFile(p);
    });

    // Missing dataset folders + likely matches (by the tag fingerprint stored in the job's LoRA)
    app.get('/api/jobs/:name/dataset-check', route((job) => {
        const missing = subsets(job).map((s, i) => ({ ...s, subset: i })).filter(s => !fs.existsSync(s.dir));
        if (!missing.length) return { missing: [] };
        const trained = trainedOn(job);
        const cands = datasetCandidates();
        return {
            missing: missing.map(s => {
                const fp = trained[path.basename(s.dir)] || Object.values(trained)[0];
                const matches = fp ? cands
                    .map(c => ({ dir: c.dir, kind: c.kind, count: c.count, score: cosine(fp.tags, c.tags) * (fp.count && c.count !== fp.count ? 0.8 : 1) }))
                    .filter(c => c.score > 0.5)
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 3) : [];
                return { subset: s.subset, dir: s.dir, trainedCount: fp?.count ?? null, matches };
            }),
        };
    }));

    // Point a subset at a found folder; an uncompiled one is compiled into <dir>/COMPILED first
    app.post('/api/jobs/:name/dataset-fix', route((job, body) => {
        let dir = body.dir;
        if (body.kind === 'uncompiled') {
            const { images, captions } = compileCategories(dir);
            dir = path.join(dir, 'COMPILED');
            fs.mkdirSync(dir, { recursive: true });
            for (const [f, src] of Object.entries(images)) {
                const to = path.join(dir, f);
                if (fs.existsSync(to)) continue;
                try { fs.linkSync(src, to); } catch { fs.copyFileSync(src, to); } // hardlink: no second copy on disk
            }
            let changed = false;
            for (const [f, text] of Object.entries(captions)) {
                const p = path.join(dir, f);
                if (fs.existsSync(p) && fs.readFileSync(p, 'utf8') === text) continue;
                fs.writeFileSync(p, text);
                changed = true;
            }
            // the trainer's text-encoder cache doesn't notice caption edits
            if (changed) fs.rmSync(path.join(dir, 'cache_text_encoder'), { recursive: true, force: true });
        }
        const p = path.join(jobDir(job), 'dataset.toml');
        const ds = TOML.parse(fs.readFileSync(p, 'utf8'));
        let i = 0;
        for (const d of [].concat(ds.datasets || [])) {
            for (const s of [].concat(d.subsets || [])) { if (i++ === body.subset) s.image_dir = dir; }
        }
        fs.writeFileSync(p, TOML.stringify(ds));
        return { dir };
    }));

    return {
        onTrainingStart(job) {
            try { snapshot(job); } catch (e) { console.error('[snapshot]', e.message); }
        },
        onTrainingEnd(job, code) {
            // wait out checkpoints()'s 15s "still being written" window so the final save is included
            setTimeout(() => kick(job, code === 0), 20000);
        },
    };
};
