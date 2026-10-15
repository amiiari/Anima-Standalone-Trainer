// Minimal TensorBoard event-file reader: scalars only (what accelerate/SummaryWriter.add_scalar writes).
// File = TFRecords: [u64 len][u32 crc][Event bytes][u32 crc]. Event: 1 wall_time(double) 2 step(int64)
// 5 summary{ 1 value{ 1 tag(string) 2 simple_value(float) } }. A torn last record (still being written) is skipped.
const fs = require('fs');
const path = require('path');

function varint(buf, pos) {
    let v = 0, mul = 1, b;
    do { b = buf[pos++]; v += (b & 0x7f) * mul; mul *= 128; } while (b & 0x80);
    return [v, pos];
}

// Calls fn(fieldNo, wireType, value, start, end) for each field in buf[start, end)
function fields(buf, start, end, fn) {
    let pos = start;
    while (pos < end) {
        let key; [key, pos] = varint(buf, pos);
        const no = Math.floor(key / 8), wt = key & 7;
        if (wt === 0) { let v; [v, pos] = varint(buf, pos); fn(no, wt, v); }
        else if (wt === 1) { fn(no, wt, buf.readDoubleLE(pos)); pos += 8; }
        else if (wt === 5) { fn(no, wt, buf.readFloatLE(pos)); pos += 4; }
        else if (wt === 2) { let len; [len, pos] = varint(buf, pos); fn(no, wt, null, pos, pos + len); pos += len; }
        else return; // groups (3/4) never appear in event files
    }
}

// -> [{ tag, step, wall, value }]
function readEventFile(file) {
    const buf = fs.readFileSync(file);
    const out = [];
    let pos = 0;
    while (pos + 12 <= buf.length) {
        const len = Number(buf.readBigUInt64LE(pos));
        const start = pos + 12, end = start + len;
        if (end + 4 > buf.length) break;
        let wall = 0, step = 0;
        const values = [];
        fields(buf, start, end, (no, wt, v, s, e) => {
            if (no === 1 && wt === 1) wall = v;
            else if (no === 2 && wt === 0) step = v;
            else if (no === 5 && wt === 2) fields(buf, s, e, (no2, wt2, _v, s2, e2) => {
                if (no2 !== 1 || wt2 !== 2) return;
                let tag = null, val = null;
                fields(buf, s2, e2, (no3, wt3, v3, s3, e3) => {
                    if (no3 === 1 && wt3 === 2) tag = buf.toString('utf8', s3, e3);
                    else if (no3 === 2 && wt3 === 5) val = v3;
                });
                if (tag !== null && val !== null) values.push({ tag, value: val });
            });
        });
        for (const v of values) out.push({ tag: v.tag, step, wall, value: v.value });
        pos = end + 4;
    }
    return out;
}

// All runs under logDir merged per tag; for a repeated step the newest write wins (resumes / restarts).
// -> { tag: [{ step, wall, value }] sorted by step }
function readScalars(logDir) {
    const files = [];
    const walk = (d) => {
        if (!fs.existsSync(d)) return;
        for (const e of fs.readdirSync(d, { withFileTypes: true })) {
            const p = path.join(d, e.name);
            if (e.isDirectory()) walk(p);
            else if (e.name.startsWith('events.out.tfevents')) files.push(p);
        }
    };
    walk(logDir);
    const byTag = {};
    for (const f of files) {
        for (const r of readEventFile(f)) {
            const m = (byTag[r.tag] ||= new Map());
            const prev = m.get(r.step);
            if (!prev || prev.wall <= r.wall) m.set(r.step, { step: r.step, wall: r.wall, value: r.value });
        }
    }
    const out = {};
    for (const [tag, m] of Object.entries(byTag)) out[tag] = [...m.values()].sort((a, b) => a.step - b.step);
    return out;
}

module.exports = { readScalars };

if (require.main === module) {
    // node lib/tbevents.js <logDir> : prints a summary per tag
    const s = readScalars(process.argv[2]);
    for (const [tag, pts] of Object.entries(s)) console.log(tag, pts.length, pts[0].step, pts[pts.length - 1].step, pts[pts.length - 1].value.toFixed(6));
}
