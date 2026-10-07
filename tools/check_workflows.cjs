const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
let checked = 0;
for (const file of fs.readdirSync(path.join(root, 'example_workflows')).filter(f => f.endsWith('.json'))) {
  const graph = JSON.parse(fs.readFileSync(path.join(root, 'example_workflows', file), 'utf8'));
  const nodes = new Map(graph.nodes.map(n => [n.id, n]));
  assert.equal(nodes.size, graph.nodes.length, 'duplicate node id');
  const links = new Map(graph.links.map(l => [l[0], l]));
  assert.equal(links.size, graph.links.length, 'duplicate link id');
  for (const [id, from, out, to, input, type] of graph.links) {
    assert(nodes.has(from) && nodes.has(to), 'missing linked node');
    const src = nodes.get(from).outputs[out], dst = nodes.get(to).inputs[input];
    assert.equal(src.type, type); assert.equal(dst.type, type);
    assert(src.links.includes(id)); assert.equal(dst.link, id);
  }
  for (const n of graph.nodes) {
    for (const i of n.inputs) if (i.link != null) assert(links.has(i.link));
    for (const o of n.outputs) for (const id of o.links || []) assert(links.has(id));
    assert.deepEqual(n.widgets_values, Array.isArray(n.widgets_values) ? Object.values(n.widgets_values_named) : n.widgets_values_named);
  }
  const pipeline = graph.nodes.find(n => n.type.startsWith('MiniMaxH3'));
  assert(pipeline);
  assert.equal(pipeline.outputs[0].type, 'IMAGE');
  assert.equal(pipeline.widgets_values_named.resolution, '480p');
  if (pipeline.type === 'MiniMaxH3V2VGenerate') {
    assert.equal(pipeline.outputs.length, 1);
    for (const key of ['pose_checkpoint', 'pose_strength', 'qwen_prompt', 'prompt_override']) assert(!(key in pipeline.widgets_values_named));
    const combine = graph.nodes.find(n => n.type === 'VHS_VideoCombine');
    if (combine) {
      assert.equal(combine.widgets_values_named.frame_rate, 24);
      assert.equal(combine.inputs.find(i => i.name === 'audio').link, null);
    }
    assert(!pipeline.inputs.some(i => i.type === 'AUDIO'));
    assert(!pipeline.outputs.some(o => o.type === 'AUDIO'));
    assert(!('h3_audio_vae' in pipeline.widgets_values_named));
    assert(pipeline.inputs.some(i => i.name === 'source_video' && i.type === 'IMAGE'));
    assert(pipeline.inputs.some(i => i.name === 'ref_image' && i.type === 'IMAGE'));
    assert.equal(pipeline.inputs.length, 2);
    const loader = graph.nodes.find(n => n.type === 'VHS_LoadVideo');
    assert(loader);
    assert.equal(loader.widgets_values_named.force_rate, 24);
    assert.equal(loader.widgets_values_named.select_every_nth, 1);
    assert.equal(loader.inputs.find(i => i.name === 'vae').link, null);
    assert(pipeline.inputs.every(i => i.link != null), 'both V2V media inputs must be connected');
    assert(!graph.nodes.some(n => /Audio|Sampler|Loader|Components/.test(n.type)));
  } else {
    assert.equal(pipeline.outputs[1].type, 'VIDEO');
    assert(pipeline.inputs.some(i => i.name === 'ref_audio' && i.type === 'AUDIO'));
    assert.equal(pipeline.widgets_values_named.ref_image_1_as_first_frame, false);
  }
  checked++; console.log('PASS ' + file + ': ' + graph.nodes.length + ' nodes, ' + graph.links.length + ' valid links');
}
const compact = fs.readFileSync(path.join(root, 'nodes_compact.py'), 'utf8');
assert(!compact.slice(compact.indexOf('class MiniMaxH3V2VGenerate')).includes('nodes_video'));
assert(compact.includes('return (images,)'));
assert(compact.includes('silent=True'));
assert(compact.includes('control_video=source'));
assert(!compact.includes('nodes_sdpose'));
assert(compact.includes('qwen_images = [source[:1], ref_image[:1]]'));
assert(!/external_pose|edit_mask|edited_first_frame/.test(compact));
assert(compact.includes('MiniMaxH3AddGuide.execute('));
const vo = fs.readFileSync(path.join(root, 'h3_prompter/video_only.py'), 'utf8');
assert(vo.includes('x[1][..., :0]'));
assert(vo.includes('torch.zeros_like(streams[1])'));
console.log('PASS static integration invariants; ' + checked + ' workflows verified. Python/GPU execution is a separate check.');
