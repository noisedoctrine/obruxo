// Exercise the actual inline application code with local DOM/download boundaries.
// This does not open a browser or substitute for a manual visual check.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function page() {
  const nodes = [], downloads = [], blobs = [];
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {}; this.paused = true; nodes.push(this); }
    append(...children) { this.children.push(...children); }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    pause() { this.paused = true; }
    click() { downloads.push({name: this.download, url: this.href}); }
  }
  const roots = {trials: new Element('div'), form: new Element('form'), status: new Element('p')};
  const packet = {packet_id: 'anonymous-packet', trials: Array.from({length: 4}, (_, index) => {
    const trial_id = `trial-${index + 1}`;
    return {trial_id, audio: Object.fromEntries(['reference', 'a', 'b'].map(slot => [slot, {file: `${trial_id}-${slot}.wav`}]))};
  })};
  const document = {
    getElementById: id => roots[id], createElement: tag => new Element(tag), createTextNode: text => ({textContent: text}),
    querySelectorAll: tag => nodes.filter(node => node.tag === tag),
  };
  class FormData {
    get(name) { return nodes.find(node => node.tag === 'select' && node.name === name).value || ''; }
  }
  const source = fs.readFileSync(path.join(__dirname, '../obruxo_training/listening.html'), 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__PACKET__', JSON.stringify(packet));
  vm.runInNewContext(script, {document, FormData, Blob,
    URL: {createObjectURL: blob => { blobs.push(blob); return 'blob:local-test'; }, revokeObjectURL() {}},
    setTimeout: callback => callback(),
  });
  return {nodes, roots, packet, downloads, blobs};
}

test('every anonymous comparison exposes three labeled audio controls and two required questions', () => {
  const {nodes, roots, packet} = page();
  assert.equal(roots.trials.children.length, packet.trials.length);
  const audio = nodes.filter(node => node.tag === 'audio');
  assert.equal(audio.length, 12);
  for (const node of audio) {
    assert.equal(node.controls, true);
    assert.equal(node.preload, 'none');
    assert.match(node.attributes['aria-label'], /^Comparison [1-4], (Reference|A|B)$/);
    assert.match(node.src, /^trial-[1-4]-(reference|a|b)\.wav$/);
  }
  const questions = nodes.filter(node => node.tag === 'select');
  assert.equal(questions.length, 8);
  assert.equal(new Set(questions.map(node => node.name)).size, 8);
  assert.ok(questions.every(node => node.required && node.children.length === 5));
});

test('starting one clip pauses every other clip', () => {
  const {nodes} = page();
  const audio = nodes.filter(node => node.tag === 'audio');
  audio.forEach(node => { node.paused = false; });
  audio[1].listeners.play();
  assert.equal(audio[1].paused, false);
  assert.ok(audio.filter(node => node !== audio[1]).every(node => node.paused));
});

test('save exports complete anonymous judgments including A, B, tie and neither', async () => {
  const {nodes, roots, packet, downloads, blobs} = page();
  const preferred = ['a', 'b', 'tie', 'neither'], usable = ['a', 'b', 'both', 'neither'];
  packet.trials.forEach((trial, index) => {
    nodes.find(node => node.name === `${trial.trial_id}-preferred`).value = preferred[index];
    nodes.find(node => node.name === `${trial.trial_id}-usable`).value = usable[index];
  });
  let prevented = false;
  roots.form.listeners.submit({target: roots.form, preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].url, 'blob:local-test');
  assert.match(downloads[0].name, /^listening-.*\.json$/);
  const result = JSON.parse(await blobs[0].text());
  assert.deepEqual(result, {format: 'blind_pairwise_listening_responses_v1', packet_id: packet.packet_id,
    judgments: packet.trials.map((trial, index) => ({trial_id: trial.trial_id, preferred: preferred[index], usable: usable[index]}))});
  assert.match(roots.status.textContent, /Judgments saved/);
});
