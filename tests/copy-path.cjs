// Executes copyImage() end to end against stubs.
//
// A syntax check cannot catch an undefined variable; only running the code
// can. This exists because two such bugs shipped: originOf() was defined
// inside render() but called from copyImage(), and a toast referenced `scale`
// where the variable is named `sc`. Both parsed cleanly and both broke copy
// for every user.
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'server/static/index.html'), 'utf8');
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];

const fn = src.match(/async function copyImage\(r\)\{[\s\S]*?\n\}/);
if (!fn) { console.error('FAIL: could not locate copyImage'); process.exit(1); }

const results = [{ full_url: 'https://example/master.jpg', thumb: 'https://wsrv.nl/?url=o',
                   width: 2988, height: 5312, license: 'CC BY 2.0' }];
let toasted = null, states = [];
const stubCard = { classList: { add: s => states.push(s), remove() {} },
                   querySelector: () => ({ currentSrc: 'https://example/rendered.jpg',
                                           classList: { add() {}, remove() {} }, textContent: '',
                                           style: {}, offsetWidth: 0 }) };
const outEl = { querySelector: () => stubCard };
const originOf = () => 'https://example/origin.jpg';
const toast = t => { toasted = t; };

globalThis.fetch = async () => ({ ok: true, headers: { get: () => 'image/jpeg' },
                                  body: null, blob: async () => ({}) });
globalThis.createImageBitmap = async () => ({ width: 2988, height: 5312 });
globalThis.document = { createElement: () => ({ getContext: () => ({ drawImage() {} }),
                                                toBlob: cb => cb({ size: 10.7e6 }) }) };
globalThis.ClipboardItem = function () {};
const clip = { write: async () => true };
Object.defineProperty(globalThis, 'navigator', { value: { clipboard: clip }, configurable: true });

(async () => {
  eval(fn[0]);
  await copyImage(results[0]);
  if (!toasted) { console.error('FAIL: copyImage produced no toast'); process.exit(1); }
  if (/failed/i.test(toasted)) { console.error('FAIL: ' + toasted); process.exit(1); }
  // 2988x5312 capped at 4096 -> 2304x4096, and the note must say capped,
  // not "largest the source offers" -- opposite meanings to the user.
  for (const want of ['2304', '4096', 'capped from 2988px', 'CC BY 2.0']) {
    if (!toasted.includes(want)) {
      console.error(`FAIL: toast missing ${want!==undefined?JSON.stringify(want):''} -> ${toasted}`);
      process.exit(1);
    }
  }
  if (!states.includes('busy') || !states.includes('copied')) {
    console.error('FAIL: progress states not set -> ' + states.join(','));
    process.exit(1);
  }
  console.log('copy path OK -> ' + toasted);
})();
