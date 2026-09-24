// Verify packaged data and exercise the shipped camera and gesture code.
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'src/app.js'), 'utf8');
const html = fs.readFileSync(process.argv[2] || path.join(root, '../docs/index.html'), 'utf8');
const selection = JSON.parse(fs.readFileSync(path.join(root, 'data/selection.json'), 'utf8'));
const catalog = JSON.parse(html.match(/<script type="application\/json" id="catalog">([\s\S]*?)<\/script>/)[1]);
const payloads = new Map([...html.matchAll(/<script type="application\/json" id="motion-([^"]+)">([\s\S]*?)<\/script>/g)].map(m => [m[1], JSON.parse(m[2])]));
const section = (start, end) => app.slice(app.indexOf(`function ${start}(`), app.indexOf(`function ${end}(`)).replace(/\s+async\s*$/, '');
const integrate = new Function(section('integrate', 'load') + ';return integrate;')();
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
assert.equal(catalog.length, 50);
assert.equal(payloads.size, 50);
assert.deepEqual(catalog.filter(m => m.section === 'continuous').map(m => m.sourceId), selection.order);
assert.equal(selection.order[0], 'H17');
assert(selection.order.slice(0, 11).every(id => id[0] === 'H'));
assert(selection.order.slice(11).every(id => id[0] === 'S'));
assert(!/id="(?:scrub|zoomIn|zoomOut|atEnd)"/.test(html));
assert(!/<(?:script|link)[^>]+(?:src|href)=/.test(html));
assert(!/__[A-Z_]+__|localhost|127\.0\.0\.1|C:\\Users|console\.log|TODO/.test(html));
const decoded = new Map();
for (const meta of catalog) {
  const payload = payloads.get(meta.id);
  const raw = zlib.inflateSync(Buffer.from(payload.blob, 'base64'));
  assert.equal(raw.length, payload.bytes);
  assert.equal(hash(raw), payload.sha256);
  const arrays = {};
  for (const [name, spec] of Object.entries(payload.layout)) {
    const T = {'<i2': Int16Array, '<f4': Float32Array, '<f8': Float64Array}[spec.type];
    assert(T, spec.type);
    const length = spec.shape.reduce((a, b) => a * b, 1);
    assert.equal(spec.offset % 8, 0);
    const bytes = raw.subarray(spec.offset, spec.offset + length * T.BYTES_PER_ELEMENT);
    arrays[name] = new T(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    assert([...arrays[name]].every(Number.isFinite));
  }
  if (arrays.humanReports) arrays.human = integrate(arrays.humanReports, meta.start, meta.humanFactor);
  arrays.paths = Array.from({length: 4}, (_, i) => integrate(meta.section === 'static'
    ? arrays['draw' + i] : arrays.reports.subarray(i * meta.duration * 2, (i + 1) * meta.duration * 2), meta.start, meta.factor));
  if (meta.section !== 'static') assert.equal(arrays.reports.length, 4 * meta.duration * 2);
  if (meta.section === 'continuous') {
    assert.equal(arrays.target.length, (meta.duration + 1) * 2);
    assert.equal(arrays.radius.length, arrays.target.length);
  }
  if (meta.section === 'renderer') assert.equal(arrays.smooth.length, arrays.humanReports.length);
  decoded.set(meta.id, arrays);
}
const cameraCheck = new Function('data', 'meta', 'panes', 'assert', 'follow', `
  const current = () => meta, isReports = () => false;
  const st = {section:'continuous', follow, zoom:1.10, pan:[0,0], t:0};
  let cameraFrames=[], viewport, sizesDirty=false;
  ${section('fullCamera', 'resize')}
  ${section('computeView', 'origin')}
  buildCameras();
  let margin=Infinity;
  for(st.t=0; st.t<=meta.duration; st.t++) {
    const view=computeView(), i=st.t*2;
    const points=[[data.target[i],data.target[i+1],data.radius[i],data.radius[i+1]],
      ...[data.human,...data.paths].filter(Boolean).map(p=>[p[i],p[i+1],0,0])];
    for(const p of panes) for(const [x,y,rx,ry] of points) {
      const gap=Math.min(p.w/2-(Math.abs(x-view.cx)+rx)*view.sx,
                         p.h/2-(Math.abs(y-view.cy)+ry)*view.sy);
      assert(gap>0, meta.id+' clipped at '+st.t+' ms');
      margin=Math.min(margin,gap);
    }
  }
  return margin;
`);
const layouts = {desktop:[[510,650],[390,300]], phone:[[340,190]], small_phone:[[270,165]], tablet:[[370,450]], landscape:[[375,310]]};
const cameraResults = [];
for (const meta of catalog.filter(m => m.section === 'continuous')) {
  const margins = {};
  for (const [name, sizes] of Object.entries(layouts)) margins[name] = +cameraCheck(decoded.get(meta.id), meta, sizes.map(([w,h]) => ({w,h,visible:true})), assert, name !== 'desktop' || meta.follow).toFixed(3);
  cameraResults.push({id:meta.id, samples:meta.duration+1, minimum_edge_margin_css_px:margins});
}
const navigationCheck = new Function('data', 'meta', 'assert', `
  const current=()=>meta, isReports=()=>false, clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const followControl={checked:true}, $=()=>followControl, markDirty=()=>{};
  const st={section:'continuous',follow:true,zoom:1.1,pan:[0,0],t:4200};
  const phone={matches:false}, defaultFollow=()=>phone.matches||meta.follow;
  const updateControls=()=>followControl.checked=st.follow;
  const listeners={}, canvas={addEventListener:(n,f)=>listeners[n]=f, setPointerCapture(){},focus(){},
    classList:{add(){},remove(){}},getBoundingClientRect:()=>({left:0,top:0})};
  const p={w:360,h:300,visible:true,canvas}, panes=[p];
  let cameraFrames=[],viewport,sizesDirty=false;
  ${section('fullCamera', 'resize')}
  ${section('computeView', 'stroke')}
  ${section('installNavigation', 'buildPanes')}
  buildCameras();computeView();installNavigation(p);
  const fire=(type,id,x,y,pointerType='touch')=>listeners[type]({type,pointerId:id,clientX:x,clientY:y,pointerType,preventDefault(){}});
  const initial=world(p,140,130);
  listeners.wheel({deltaY:-80,clientX:140,clientY:130,preventDefault(){}});
  assert(st.follow && followControl.checked,'Wheel zoom disabled following');
  computeView(); const anchored=world(p,140,130);
  assert(Math.hypot(...initial.map((v,i)=>v-anchored[i]))<1e-8,'Wheel anchor moved');
  const zoomed=st.zoom;
  listeners.wheel({deltaY:80,clientX:140,clientY:130,preventDefault(){}});
  assert(st.follow && st.zoom<zoomed,'Wheel unzoom disabled following');
  fire('pointerdown',1,100,100);fire('pointerdown',2,200,100);
  fire('pointermove',2,230,100);fire('pointerup',2,230,100);fire('pointerup',1,100,100);
  assert(st.follow && followControl.checked,'Pinch disabled following');
  fire('pointerdown',3,100,100);fire('pointermove',3,120,115);fire('pointerup',3,120,115);
  assert(!st.follow && !followControl.checked,'Touch pan did not disable following');
  fit();assert(st.follow,'Reset did not restore default follow');
  fire('pointerdown',4,100,100,'mouse');assert(!st.follow,'Mouse click did not disable following');
  fire('pointermove',4,130,120,'mouse');fire('pointerup',4,130,120,'mouse');
  listeners.wheel({deltaY:-80,clientX:140,clientY:130,preventDefault(){}});
  assert(!st.follow,'Zoom unexpectedly re-enabled following after a pan');
  fit();fire('pointerdown',5,100,100);fire('pointerup',5,100,100);
  assert(!st.follow,'Single touch tap did not disable following');
  phone.matches=true; fit(); fire('pointerdown',6,100,100);fire('pointermove',6,140,120);fire('pointerup',6,140,120);
  assert(st.follow,'Phone touch should keep following by default');
  fire('pointerdown',7,100,100);fire('pointerdown',8,200,100);fire('pointermove',8,240,100);fire('pointerup',8,240,100);fire('pointerup',7,100,100);
  assert(st.follow,'Phone pinch should keep following');
  st.follow=false;updateControls();const oldPan=st.pan[0];
  fire('pointerdown',9,100,100);fire('pointermove',9,130,100);fire('pointerup',9,130,100);
  assert(!st.follow && st.pan[0]!==oldPan,'Phone manual pan should work when following is disabled in View');
  return ['wheel-follow','wheel-anchor','unzoom-follow','pinch-follow','touch-pan','mouse-pan','manual-zoom','tap','reset','phone-follow','phone-pinch','phone-manual'];
`);
const h17 = catalog.find(m => m.id === 'continuous/H17');
const navigation = navigationCheck(decoded.get(h17.id), h17, assert);
new Function('assert', `
  const st={playing:true,t:0,speed:1}; const data={};
  const current=()=>({duration:8000}), dialogOpen=()=>false, draw=()=>{}, requestAnimationFrame=()=>1;
  const document={hidden:false}; let raf=0,last=null,ending=0,dirty=true;
  ${section('frame','fullCamera')}
  frame(0); frame(1000); assert.equal(st.t,1000,'Presentation gaps must not slow the clock');
  frame(8000); assert.equal(st.t,8000); frame(8600); assert.equal(st.t,8000);
  frame(8700); assert.equal(st.t,0,'Playback should repeat automatically');
`)(assert);
const receipt = {html_sha256:hash(Buffer.from(html)),examples:catalog.length, data_checks:'passed',camera:cameraResults,navigation,clock:['elapsed-time','automatic-repeat']};
fs.writeFileSync(path.join(__dirname, 'verification.json'),JSON.stringify(receipt,null,2)+'\n');
console.log('PASS: 50 data payloads; all 18 Continuous motions at every millisecond across five layouts; 12 gesture checks; real-time clock and automatic repeat.');
