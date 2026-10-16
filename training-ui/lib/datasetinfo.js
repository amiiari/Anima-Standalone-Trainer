// Dataset facts for the Dataset tab: image sizes (read from file headers, no decoding) and the
// resolution / bucket settings they suggest.
const fs = require('fs');
const path = require('path');

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

// -> [width, height] or null. PNG IHDR, JPEG SOFn, WEBP VP8/VP8L/VP8X.
function imageSize(file) {
    const fd = fs.openSync(file, 'r');
    try {
        const head = Buffer.alloc(64);
        fs.readSync(fd, head, 0, 64, 0);
        if (head.readUInt32BE(0) === 0x89504e47) return [head.readUInt32BE(16), head.readUInt32BE(20)];
        if (head.toString('ascii', 0, 4) === 'RIFF' && head.toString('ascii', 8, 12) === 'WEBP') {
            const kind = head.toString('ascii', 12, 16);
            if (kind === 'VP8 ') return [head.readUInt16LE(26) & 0x3fff, head.readUInt16LE(28) & 0x3fff];
            if (kind === 'VP8L') { const b = head.readUInt32LE(21); return [(b & 0x3fff) + 1, ((b >> 14) & 0x3fff) + 1]; }
            if (kind === 'VP8X') return [head.readUIntLE(24, 3) + 1, head.readUIntLE(27, 3) + 1];
            return null;
        }
        if (head[0] === 0xff && head[1] === 0xd8) {
            // walk JPEG segments to the first SOFn (C0-CF, minus DHT C4 / JPG C8 / DAC CC)
            const size = fs.fstatSync(fd).size;
            const seg = Buffer.alloc(9);
            let pos = 2;
            while (pos + 9 <= size) {
                fs.readSync(fd, seg, 0, 9, pos);
                if (seg[0] !== 0xff) return null;
                const marker = seg[1];
                if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return [seg.readUInt16BE(7), seg.readUInt16BE(5)];
                pos += 2 + seg.readUInt16BE(2);
            }
        }
        return null;
    } finally {
        fs.closeSync(fd);
    }
}

// Forge/A1111 generation settings stored in a PNG ("parameters" tEXt/iTXt/zTXt chunk, before the image data).
// -> { infotext, negative, width, height, hrScale } or null
function forgeSettings(file) {
    if (!/\.png$/i.test(file)) return null;
    const fd = fs.openSync(file, 'r');
    let text = null;
    try {
        const hdr = Buffer.alloc(8);
        let pos = 8;
        while (fs.readSync(fd, hdr, 0, 8, pos) === 8) {
            const len = hdr.readUInt32BE(0), type = hdr.toString('ascii', 4, 8);
            if (type === 'IDAT' || type === 'IEND') break;
            if (type === 'tEXt' || type === 'iTXt' || type === 'zTXt') {
                const data = Buffer.alloc(len);
                fs.readSync(fd, data, 0, len, pos + 8);
                const nul = data.indexOf(0);
                if (data.toString('latin1', 0, nul) === 'parameters') {
                    if (type === 'tEXt') text = data.toString('latin1', nul + 1);
                    else if (type === 'zTXt') text = require('zlib').inflateSync(data.subarray(nul + 2)).toString('latin1');
                    else { // iTXt: flag, method, language\0, translated keyword\0, text
                        const compressed = data[nul + 1] === 1;
                        let p = data.indexOf(0, nul + 3) + 1;
                        p = data.indexOf(0, p) + 1;
                        const body = data.subarray(p);
                        text = (compressed ? require('zlib').inflateSync(body) : body).toString('utf8');
                    }
                    break;
                }
            }
            pos += 12 + len;
        }
    } finally {
        fs.closeSync(fd);
    }
    if (!text || !/\bSteps: /.test(text)) return null;
    const lines = text.split('\n');
    const last = lines[lines.length - 1];
    const neg = text.match(/\nNegative prompt: ([\s\S]*?)\n(?=[^\n]*\bSteps: )/);
    const size = last.match(/\bSize: (\d+)x(\d+)/);
    const hr = last.match(/\bHires upscale: ([\d.]+)/);
    return {
        infotext: text,
        negative: neg ? neg[1] : '',
        width: size ? +size[1] : null,
        height: size ? +size[2] : null,
        hrScale: hr ? +hr[1] : 1,
    };
}

const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const down64 = (v) => Math.floor(v / 64) * 64;

// Settings that fit the images (same shape as picked by hand for jennie/reiqun):
//   resolution  = long side reached by 75% of images, capped (1216 = the cap used for large datasets), >= 512
//   max bucket  = resolution (longest side never goes past it)
//   min bucket  = resolution x the narrowest short/long ratio (5th percentile, so one odd image doesn't decide it)
//   no upscale  = on when some images are smaller than the resolution (don't train on blown-up images)
function suggestBuckets(sizes, cap = 1216) {
    if (!sizes.length) return null;
    const longs = sizes.map(([w, h]) => Math.max(w, h)).sort((a, b) => a - b);
    const ratios = sizes.map(([w, h]) => Math.min(w, h) / Math.max(w, h)).sort((a, b) => a - b);
    const resolution = Math.max(512, Math.min(cap, down64(pct(longs, 0.25))));
    return {
        resolution,
        min_bucket_reso: Math.max(256, down64(resolution * pct(ratios, 0.05))),
        max_bucket_reso: resolution,
        bucket_no_upscale: longs[0] < resolution,
    };
}

// subsets: [{ dir, repeats, ext }] -> per-subset and overall facts + suggested settings
function datasetInfo(subsets, resolution) {
    const all = [];
    const tags = new Map();
    let captions = 0, tagCount = 0, perEpoch = 0;
    const out = subsets.map(({ dir, repeats = 1, ext = '.txt' }) => {
        const s = { dir, repeats, exists: !!dir && fs.existsSync(dir), images: 0, missingCaptions: [], unreadable: [] };
        if (!s.exists) return s;
        // trigger.txt sits in the dataset version folder (COMPILED's parent) or the folder itself
        const tf = [path.join(dir, 'trigger.txt'), path.join(dir, '..', 'trigger.txt')].find(p => fs.existsSync(p));
        s.trigger = tf ? fs.readFileSync(tf, 'utf8').trim() : null;
        s.withTrigger = 0;
        for (const f of fs.readdirSync(dir)) {
            if (!IMAGE_EXT.test(f)) continue;
            s.images++;
            let size = null;
            try { size = imageSize(path.join(dir, f)); } catch { }
            if (size) all.push(size); else s.unreadable.push(f);
            const cp = path.join(dir, f.replace(/\.[^.]+$/, ext));
            if (!fs.existsSync(cp)) { s.missingCaptions.push(f); continue; }
            const text = fs.readFileSync(cp, 'utf8');
            const t = text.split(',').map(x => x.trim()).filter(Boolean);
            captions++;
            tagCount += t.length;
            for (const tag of t) tags.set(tag, (tags.get(tag) || 0) + 1);
            if (s.trigger && t[0] === s.trigger) s.withTrigger++;
        }
        perEpoch += s.images * repeats;
        return s;
    });
    const areas = all.map(([w, h]) => w * h).sort((a, b) => a - b);
    const ratios = all.map(([w, h]) => w / h).sort((a, b) => a - b);
    return {
        subsets: out,
        images: all.length,
        imagesPerEpoch: perEpoch,
        medianSize: areas.length ? Math.round(Math.sqrt(pct(areas, 0.5))) : null,
        smallestSide: all.length ? Math.min(...all.map(([w, h]) => Math.min(w, h))) : null,
        aspect: ratios.length ? [pct(ratios, 0.05), pct(ratios, 0.95)] : null,
        upscaled: resolution ? all.filter(([w, h]) => Math.max(w, h) < resolution).length : null,
        captions,
        uniqueTags: tags.size,
        avgTags: captions ? tagCount / captions : 0,
        suggested: suggestBuckets(all),
    };
}

module.exports = { imageSize, forgeSettings, suggestBuckets, datasetInfo };

if (require.main === module) {
    // node lib/datasetinfo.js : checks the bucket rule against the hand-picked settings it was based on
    const assert = require('assert');
    const jennieLike = [[1280, 1856], [1856, 1280], [1344, 1792], [1400, 1800], [1500, 1500]];
    assert.deepStrictEqual(suggestBuckets(jennieLike), { resolution: 1216, min_bucket_reso: 832, max_bucket_reso: 1216, bucket_no_upscale: false });
    const small = [[512, 768], [768, 1024], [896, 896], [1024, 768], [700, 1050]];
    const s = suggestBuckets(small);
    assert.ok(s.resolution === 896 && s.bucket_no_upscale, JSON.stringify(s)); // 75% of these reach 896px
    // a PNG header
    const tmp = path.join(require('os').tmpdir(), 'dsinfo-test.png');
    const png = Buffer.alloc(33);
    png.writeUInt32BE(0x89504e47, 0); png.writeUInt32BE(0x0d0a1a0a, 4); png.writeUInt32BE(13, 8); png.write('IHDR', 12);
    png.writeUInt32BE(832, 16); png.writeUInt32BE(1216, 20);
    fs.writeFileSync(tmp, png);
    assert.deepStrictEqual(imageSize(tmp), [832, 1216]);
    fs.unlinkSync(tmp);
    console.log('datasetinfo OK');
}
