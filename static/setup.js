const $ = id => document.getElementById(id);

const areaSelect = $('areaSelect');
const modeSelect = $('modeSelect');
const snapImg    = $('snapImg');
const overlay    = $('overlay');
const ctx        = overlay.getContext('2d');
const selInfo    = $('selInfo');
const btnSave    = $('btnSave');
const btnClear   = $('btnClear');

let config = {areas: []};
let selection = null;           // {x0,y0,x1,y1} in canvas pixels
let nat = {w:0,h:0};            // image natural size
let snapshotReady = false;

// ---------- wiring ----------
$('btnSnap').onclick = takeSnapshot;
$('btnSave').onclick = saveSelection;
$('btnClear').onclick = clearSelection;

window.addEventListener('load', async ()=>{
  await loadConfig();
  populateAreas();
  snapImg.addEventListener('load', ()=>{
    nat.w = snapImg.naturalWidth;
    nat.h = snapImg.naturalHeight;
    sizeCanvasToImage();
    snapshotReady = true;
  });
  window.addEventListener('resize', sizeCanvasToImage);
  enableDrawing();
});

// ---------- backend I/O ----------
async function loadConfig(){
  const res = await fetch('/api/config');
  config = await res.json();
}

async function saveConfig(){
  const res = await fetch('/api/config', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ areas: config.areas || [] })
  });
  const js = await res.json();
  if(!js.ok){ throw new Error(js.error || 'Save failed'); }
}

function populateAreas(){
  areaSelect.innerHTML = '';
  for (const a of (config.areas || [])) {
    const opt = document.createElement('option');
    opt.value = a.name;
    opt.textContent = a.name;
    areaSelect.appendChild(opt);
  }
  renderProfiles();
}

// Fetch raw JPEG and show it
async function takeSnapshot(){
  const res = await fetch('/api/snapshot', { method:'POST' });
  if(!res.ok){ alert('Snapshot failed'); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  snapImg.src = url;
}

// ---------- canvas / selection ----------
function sizeCanvasToImage(){
  overlay.width  = snapImg.clientWidth;
  overlay.height = snapImg.clientHeight;
  drawSelection();
}

function enableDrawing(){
  overlay.addEventListener('mousedown', e=>{
    if(!snapshotReady) return;
    const r = overlay.getBoundingClientRect();
    selection = {x0:e.clientX-r.left, y0:e.clientY-r.top, x1:e.clientX-r.left, y1:e.clientY-r.top};
    drawSelection(); updateSelInfo(); btnSave.disabled=false; btnClear.disabled=false;
  });
  window.addEventListener('mousemove', e=>{
    if(!selection) return;
    const r = overlay.getBoundingClientRect();
    selection.x1 = Math.min(Math.max(e.clientX-r.left, 0), overlay.width);
    selection.y1 = Math.min(Math.max(e.clientY-r.top , 0), overlay.height);
    drawSelection(); updateSelInfo();
  });
  window.addEventListener('mouseup', ()=>{ /* keep selection */ });
}

function drawSelection(){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!selection) return;
  const x = Math.min(selection.x0, selection.x1);
  const y = Math.min(selection.y0, selection.y1);
  const w = Math.abs(selection.x1 - selection.x0);
  const h = Math.abs(selection.y1 - selection.y0);
  ctx.strokeStyle = '#61dafb';
  ctx.lineWidth = 2;
  ctx.strokeRect(x,y,w,h);
  ctx.fillStyle = 'rgba(97,218,251,0.2)';
  ctx.fillRect(x,y,w,h);
}

function updateSelInfo(){
  if(!selection){ selInfo.textContent = 'No selection.'; return; }
  const rect = canvasToPixels(selection);
  selInfo.textContent = `Selection (px): [${rect.x}, ${rect.y}, ${rect.w}, ${rect.h}]`;
}

function canvasToPixels(sel){
  const sx = nat.w / overlay.width;
  const sy = nat.h / overlay.height;
  const x = Math.round(Math.min(sel.x0, sel.x1) * sx);
  const y = Math.round(Math.min(sel.y0, sel.y1) * sy);
  const w = Math.round(Math.abs(sel.x1 - sel.x0) * sx);
  const h = Math.round(Math.abs(sel.y1 - sel.y0) * sy);
  return {x,y,w,h};
}

function clearSelection(){
  selection = null; drawSelection(); updateSelInfo();
  btnSave.disabled = true; btnClear.disabled = true;
}

// ---------- color profile math (client-side) ----------
// Convert RGB(0..255) -> HSV(OpenCV ranges: H 0..179, S 0..255, V 0..255)
function rgb2hsv_cv(r,g,b){
  r/=255; g/=255; b/=255;
  const max = Math.max(r,g,b), min = Math.min(r,g,b);
  let h, s, v = max;
  const d = max - min;
  s = max === 0 ? 0 : d / max;
  if(max === min){ h = 0; }
  else{
    switch(max){
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4;
    }
    h /= 6;
  }
  // map to OpenCV ranges
  return [Math.round(h*179), Math.round(s*255), Math.round(v*255)];
}

function computeHSVStatsFromRect(rectPx){
  // draw the selected region of the image onto an offscreen canvas to read pixels
  const off = document.createElement('canvas');
  off.width = rectPx.w; off.height = rectPx.h;
  const octx = off.getContext('2d');
  // draw src image scaled 1:1 for the region
  // Create another canvas with the full snapshot at natural size to avoid scaling artifacts
  const full = document.createElement('canvas');
  full.width = nat.w; full.height = nat.h;
  const fctx = full.getContext('2d');
  fctx.drawImage(snapImg, 0, 0, nat.w, nat.h);
  const imgData = fctx.getImageData(rectPx.x, rectPx.y, rectPx.w, rectPx.h);
  const data = imgData.data;

  let n = rectPx.w * rectPx.h;
  if(n <= 0) return {mean:[0,0,0], std:[0,0,0]};

  // accumulate
  let sum = [0,0,0];
  let sum2 = [0,0,0];
  for(let i=0;i<data.length;i+=4){
    const r=data[i], g=data[i+1], b=data[i+2];
    const [h,s,v] = rgb2hsv_cv(r,g,b);
    sum[0]+=h; sum[1]+=s; sum[2]+=v;
    sum2[0]+=h*h; sum2[1]+=s*s; sum2[2]+=v*v;
  }
  const mean = sum.map(x=>x/n);
  const std  = sum2.map((x,i)=>Math.sqrt(Math.max(0, x/n - mean[i]*mean[i])));

  return {mean, std};
}

function deriveWindow(mean,std){
  const hPad = Math.max(8, 1.5*std[0]);
  const sPad = Math.max(40, 2.0*std[1]);
  const vPad = Math.max(40, 2.0*std[2]);
  const lo = [Math.max(0,   Math.round(mean[0]-hPad)),
              Math.max(0,   Math.round(mean[1]-sPad)),
              Math.max(0,   Math.round(mean[2]-vPad))];
  const hi = [Math.min(179, Math.round(mean[0]+hPad)),
              Math.min(255, Math.round(mean[1]+sPad)),
              Math.min(255, Math.round(mean[2]+vPad))];
  return {lo,hi,mean,std};
}

// ---------- Save ----------
async function saveSelection(){
  if(!selection || !areaSelect.value) return;
  const rect = canvasToPixels(selection);
  const mode = modeSelect.value;

  await loadConfig(); // refresh latest
  const idx = (config.areas || []).findIndex(a => a.name === areaSelect.value);
  if(idx === -1){ alert('Area not found'); return; }

  if(mode === 'area'){
    // Set ROI in pixels (your detection loop expects pixel rects)
    config.areas[idx].rect = [rect.x, rect.y, rect.w, rect.h];
  }else{
    // Compute HSV window client-side and append to profiles
    const {mean,std} = computeHSVStatsFromRect(rect);
    const {lo,hi} = deriveWindow(mean,std);
    if(!config.areas[idx].profiles) config.areas[idx].profiles = [];
    config.areas[idx].profiles.push({
      hsv_lo: lo, hsv_hi: hi,
      label: `auto ${new Date().toISOString().slice(0,19).replace('T',' ')}`
    });
  }

  await saveConfig();
  await loadConfig();
  populateAreas();
}
