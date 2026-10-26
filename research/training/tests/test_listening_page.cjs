// Exercise the actual inline application code with local DOM/download boundaries.
// This does not open a browser or substitute for a manual visual check.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function page({storage = new Map(), initialAnswers = {}, condition} = {}) {
  const nodes = [], downloads = [], blobs = [];
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {}; this.paused = true; this.currentTime = 0; this.controls = false; this.playCount = 0; nodes.push(this); }
    append(...children) { this.children.push(...children); }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    pause() { this.paused = true; }
    play() { this.paused = false; this.playCount++; return Promise.resolve(); }
    click() { if (this.listeners.click) return this.listeners.click(); downloads.push({name: this.download, url: this.href}); }
  }
  const roots = {trials: new Element('div'), form: new Element('form'), status: new Element('p'), 'condition-note': new Element('p')};
  const packet = {packet_id: 'anonymous-packet', condition, trials: Array.from({length: 4}, (_, index) => {
    const trial_id = `trial-${index + 1}`;
    return {trial_id, audio: Object.fromEntries(['reference', 'a', 'b'].map(slot => [slot, {file: `${trial_id}-${slot}.wav`}]))};
  })};
  const document = {
    getElementById: id => roots[id], createElement: tag => new Element(tag), createTextNode: text => ({textContent: text}),
    querySelectorAll: tag => nodes.filter(node => node.tag === tag),
  };
  class FormData {
    get(name) { return nodes.find(node => node.tag === 'input' && node.name === name && node.checked)?.value || null; }
  }
  const source = fs.readFileSync(path.join(__dirname, '../obruxo_training/listening.html'), 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__PACKET__', JSON.stringify(packet)).replace('__INITIAL_ANSWERS__', JSON.stringify(initialAnswers));
  vm.runInNewContext(script, {document, FormData, Blob,
    URL: {createObjectURL: blob => { blobs.push(blob); return 'blob:local-test'; }, revokeObjectURL() {}},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)}, setTimeout: callback => callback(),
  });
  const choose = (name, value) => nodes.filter(node => node.tag === 'input' && node.name === name).forEach(node => { node.checked = node.value === value; });
  return {nodes, roots, packet, downloads, blobs, choose, storage};
}

test('every comparison preloads its clips and exposes all answers as required radio pills', () => {
  const {nodes, roots, packet} = page();
  assert.equal(roots.trials.children.length, packet.trials.length);
  const audio = nodes.filter(node => node.tag === 'audio');
  assert.equal(audio.length, 12);
  for (const node of audio) {
    assert.equal(node.controls, false);
    assert.equal(node.preload, 'auto');
    assert.match(node.attributes['aria-label'], /^Comparison [1-4], (Reference|A|B)$/);
    assert.match(node.src, /^trial-[1-4]-(reference|a|b)\.wav$/);
  }
  assert.equal(nodes.filter(node => node.tag === 'select').length, 0);
  const answers = nodes.filter(node => node.tag === 'input');
  assert.equal(answers.length, 32);
  assert.equal(new Set(answers.map(node => node.name)).size, 8);
  assert.ok(answers.every(node => node.required && node.type === 'radio'));
});

test('every play click restarts immediately, including the already playing clip, and stops other clips', () => {
  const {nodes} = page();
  const audio = nodes.filter(node => node.tag === 'audio');
  const buttons = nodes.filter(node => node.className === 'play');
  audio[0].currentTime = 2.5;
  buttons[0].click();
  assert.equal(audio[0].currentTime, 0);
  audio[0].currentTime = 1.2;
  buttons[0].click();
  assert.equal(audio[0].currentTime, 0);
  assert.equal(audio[0].playCount, 2);
  buttons[1].click();
  assert.equal(audio[1].paused, false);
  assert.ok(audio.filter(node => node !== audio[1]).every(node => node.paused));
});

test('save exports complete anonymous judgments including A, B, tie and neither', async () => {
  const {roots, packet, downloads, blobs, choose} = page();
  const preferred = ['a', 'b', 'tie', 'neither'], usable = ['a', 'b', 'both', 'neither'];
  packet.trials.forEach((trial, index) => {
    choose(`${trial.trial_id}-preferred`, preferred[index]);
    choose(`${trial.trial_id}-usable`, usable[index]);
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

test('existing choices survive the page upgrade and later browser reloads', () => {
  const initialAnswers = {'trial-1-preferred': 'b', 'trial-1-usable': 'neither'};
  const first = page({initialAnswers});
  assert.equal(first.nodes.find(node => node.name === 'trial-1-preferred' && node.checked).value, 'b');
  first.choose('trial-1-preferred', 'tie');
  first.roots.form.listeners.change({currentTarget: first.roots.form});
  const reloaded = page({storage: first.storage, initialAnswers});
  assert.equal(reloaded.nodes.find(node => node.name === 'trial-1-preferred' && node.checked).value, 'tie');
  assert.equal(reloaded.nodes.find(node => node.name === 'trial-1-usable' && node.checked).value, 'neither');
});

test('tone and original-level tasks are labeled as separate listening conditions', () => {
  assert.match(page({condition: {purpose: 'timbre'}}).roots['condition-note'].textContent, /Volume matched/);
  assert.match(page({condition: {purpose: 'level_accuracy'}}).roots['condition-note'].textContent, /judge.*reference level/);
  assert.match(page().roots['condition-note'].textContent, /unsuitable for ranking tone/);
});
