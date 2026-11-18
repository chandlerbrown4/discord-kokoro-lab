"""Local microphone comparison page. All assets are served by vc-studio."""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microphone comparison · Voice Lab</title>
<style>
:root{color-scheme:light;--ink:#152623;--muted:#506661;--border:#cbd7d2;--accent:#136551;--paper:#fdfefc}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#edf2ec;color:var(--ink);font:16px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header,main{max-width:760px;margin:auto;padding:0 24px}header{display:flex;align-items:center;gap:10px;height:76px;font-size:17px;font-weight:700}
header span{height:22px;width:23px;background:linear-gradient(90deg,transparent 0 5%,#136551 5% 18%,transparent 18% 31%,#136551 31% 45%,transparent 45% 56%,#136551 56% 70%,transparent 70% 82%,#136551 82% 95%,transparent 95%);clip-path:polygon(0 39%,23% 39%,23% 12%,50% 12%,50% 0,76% 0,76% 28%,100% 28%,100% 72%,76% 72%,76% 100%,50% 100%,50% 88%,23% 88%,23% 61%,0 61%)}
main{padding-bottom:50px}h1{font-size:clamp(2.3rem,5vw,3.55rem);line-height:1.08;letter-spacing:-.055em;margin:16px 0 12px}p{margin:0;color:var(--muted)}.intro{margin-bottom:26px}
section{background:var(--paper);border:1px solid var(--border);border-radius:14px;padding:clamp(17px,4vw,30px);margin:14px 0}
h2{font-size:1.15rem;letter-spacing:-.025em;margin:0 0 16px}.row{display:flex;align-items:end;gap:12px;flex-wrap:wrap}.row label{min-width:0;flex:1}.row label:last-child{flex:0 0 98px}
label{display:block;color:var(--muted);font-size:.88rem;font-weight:600}select,button,.file-label{font:inherit;border:1px solid var(--border);border-radius:8px;min-height:42px;padding:8px 12px}
select{margin-top:4px;width:100%;color:var(--ink);background:white}button,.file-label{cursor:pointer;background:#fff;color:var(--ink);font-weight:600;font-size:.93rem}
button:disabled,select:disabled{opacity:.55;cursor:default}button:focus-visible,input:focus-visible+label,select:focus-visible,a:focus-visible{outline:3px solid #458874;outline-offset:3px}
.primary{background:var(--accent);color:white;border-color:var(--accent)}.recording{background:#a43624;border-color:#a43624}
.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}.meter{background:#e3ebe6;height:7px;overflow:hidden;border-radius:9px;margin:10px 0}
.meter i{display:block;width:0;height:100%;background:#26866e;transition:width .08s linear}.status{font-size:.9rem;min-height:1.5em}.error{color:#a43624}
.drop{border-top:1px solid var(--border);margin-top:18px;padding-top:20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}.drag .drop{background:#e1f0e8}.drop p,.meta{font-size:.87rem}.file-label{display:inline-flex;align-items:center;white-space:nowrap}
#file{position:absolute;width:1px;height:1px;opacity:0}.busy .file-label{opacity:.55;cursor:default}
audio{display:block;width:100%;margin:18px 0 10px}a{color:var(--accent);font-weight:600}.result-top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.result-top h2{margin:0}.result-top a{font-size:.88rem}
@media(max-width:520px){header{height:58px}header,main{padding-left:15px;padding-right:15px}.row label{flex-basis:100%}.row label:last-child{flex:1}.drop{gap:8px}}
</style></head><body>
<header><span aria-hidden="true"></span>Voice Lab</header>
<main><h1>Microphone comparison</h1><p class="intro">Hear your microphone take, then the same words in the converted voice. Audio is processed on this computer.</p>
<section id="inputBox"><h2>Record a take</h2>
<div class="row"><label>Microphone<select id="mic"><option value="">System default</option></select></label><label>Length<select id="seconds">
<option value="3">3 sec</option><option value="5">5 sec</option><option value="8" selected>8 sec</option><option value="10">10 sec</option></select></label></div>
<div class="actions"><button id="start" class="primary">Enable microphone</button><button id="off" hidden>Turn mic off</button></div>
<div class="meter" aria-hidden="true"><i id="level"></i></div><p class="status" role="status" id="status">Microphone is off.</p>
<div class="drop"><input id="file" type="file" accept=".wav,.webm,.ogg,.m4a,.mp4,.mp3,.flac,.aac,.aiff,.aif">
<label class="file-label" for="file">Choose audio file</label><p>Or drop a spoken take here to convert it. Up to 30 seconds.</p></div></section>
<section><div class="result-top"><h2>Your comparison</h2><a id="save" download="side-by-side.wav" hidden>Save WAV</a></div>
<audio id="player" controls hidden></audio><p id="result" class="meta">A completed take saves the dry, converted and paired WAVs in your local output folder.</p></section></main>
<script>
"use strict";
const el = id => document.getElementById(id);
const mic=el("mic"), start=el("start"), off=el("off"), picker=el("file"), status=el("status"), box=el("inputBox");
let stage="idle", stream, context, animation, recorder, timer, countdown, started, url, mode, permissionVersion=0;
const types=["audio/webm;codecs=opus","audio/mp4","audio/ogg;codecs=opus","audio/webm"];
const fileTypes={wav:"audio/wav",mp3:"audio/mpeg",webm:"audio/webm",ogg:"audio/ogg",m4a:"audio/mp4",mp4:"audio/mp4",flac:"audio/flac",aac:"audio/aac",aif:"audio/aiff",aiff:"audio/aiff"};
function message(text,error=false){status.textContent=text;status.classList.toggle("error",error);}
function state(next){
  stage=next;
  const busy=["opening","recording","converting"].includes(stage);
  picker.disabled=busy; box.classList.toggle("busy",busy);
  mic.disabled=busy; el("seconds").disabled=busy;
  start.disabled=["opening","recording","converting"].includes(stage);
  start.textContent=({idle:"Enable microphone",opening:"Opening microphone…",ready:"Record take",recording:"Recording…",converting:"Converting…"})[stage];
  start.classList.toggle("recording",stage==="recording"); off.hidden=stage!=="ready";
}
function release(){
  clearTimeout(timer);clearInterval(countdown);cancelAnimationFrame(animation);
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  if(context){context.close().catch(()=>{});context=null;}el("level").style.width="0";
}
function idle(text,error=false){permissionVersion++;release();state("idle");message(text,error);}
async function enable(){
  const version=++permissionVersion;
  state("opening");message("Allow microphone access in this browser.");
  try{
    if(!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw Error("This browser does not support recording here. Choose an audio file below.");
    mode=types.find(t=>MediaRecorder.isTypeSupported(t));
    if(!mode) throw Error("This browser has no supported recording format. Choose an audio file below.");
    const input=mic.value ? {deviceId:{exact:mic.value}} : {};
    const opened=await navigator.mediaDevices.getUserMedia({audio:{...input,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    if(version!==permissionVersion){opened.getTracks().forEach(t=>t.stop());return;}
    stream=opened;
    const device=stream.getAudioTracks()[0], selected=device.getSettings().deviceId;
    const names=await navigator.mediaDevices.enumerateDevices();
    if(version!==permissionVersion)return;
    mic.replaceChildren(new Option("System default",""));
    names.filter(d=>d.kind==="audioinput").forEach((d,i)=>mic.add(new Option(d.label||("Microphone "+(i+1)),d.deviceId)));
    mic.value=selected||"";
    const AC=window.AudioContext||window.webkitAudioContext;
    context=new AC(); const analyser=context.createAnalyser(); analyser.fftSize=1024;
    context.createMediaStreamSource(stream).connect(analyser); await context.resume();
    if(version!==permissionVersion)return;
    const vals=new Float32Array(analyser.fftSize);
    function meter(){analyser.getFloatTimeDomainData(vals);let max=0;for(const v of vals)max=Math.max(max,Math.abs(v));
      el("level").style.width=(100*Math.max(0,Math.min(1,(20*Math.log10(max||1e-8)+65)/65)))+"%";animation=requestAnimationFrame(meter);}
    meter();state("ready");message("Mic is on. Speak and check that the level moves before recording.");
    device.onended=()=>{if(stage==="ready") idle("The selected microphone disconnected.",true);};
  }catch(e){if(version===permissionVersion)idle(e.name==="NotAllowedError"?"Microphone access was denied. Enable it in browser settings or choose a file.":e.message,true);}
}
async function send(blob,type){
  if(!blob.size){idle("The take contains no audio. Record again or choose a file.",true);return;}
  release();state("converting");message("Converting your take on this computer…");
  try{
    const r=await fetch("/take",{method:"POST",headers:{"Content-Type":type},body:blob});
    if(!r.ok) throw Error((await r.text()).trim().slice(0,400)||"Conversion failed.");
    if(r.headers.get("Content-Type")?.split(";")[0]!=="audio/wav")throw Error("The server returned an unexpected audio format.");
    const wav=await r.blob();if(!wav.size)throw Error("The server returned an empty comparison.");
    const next=URL.createObjectURL(wav), previous=url;url=next;
    el("player").src=next;el("player").hidden=false;el("save").href=next;el("save").hidden=false;if(previous)URL.revokeObjectURL(previous);
    const folder=r.headers.get("X-Preview-Take");el("result").textContent="Dry microphone first, converted voice second. "+(folder?"Saved in "+folder+" in your local output folder.":"Saved in your local output folder.");
    idle("Ready to listen. The microphone is off.");
  }catch(e){idle(e.message,true);}
}
function record(){
  if(!stream || !stream.getAudioTracks().some(t=>t.readyState==="live")){idle("Enable a microphone to record.");return;}
  const seconds=Number(el("seconds").value),chunks=[];
  try{
    const take=new MediaRecorder(stream,{mimeType:mode});recorder=take;
    take.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    take.onerror=e=>{recorder=null;take.onstop=null;idle(e.error?.message||"Recording failed.",true);};
    take.onstop=()=>{if(recorder!==take)return;recorder=null;const type=take.mimeType||mode;send(new Blob(chunks,{type}),type);};
    take.start();started=performance.now();state("recording");message("Recording now. Speak into your microphone.");
    countdown=setInterval(()=>{start.textContent="Recording · "+Math.max(0,Math.ceil(seconds-(performance.now()-started)/1000))+" s";},150);
    timer=setTimeout(()=>{if(take.state!=="inactive")take.stop();},seconds*1000);
  }catch(e){recorder=null;idle(e.message,true);}
}
function choose(file){
  if(!file||!["idle","ready"].includes(stage))return;
  const ext=file.name.split(".").pop().toLowerCase(), type=fileTypes[ext]||file.type.split(";")[0].toLowerCase();
  if(!Object.values(fileTypes).includes(type)){idle("Choose a WAV, WebM, Ogg, M4A, MP3, FLAC, AAC or AIFF audio file.",true);return;}
  send(file,type);
}
start.onclick=()=>{if(stage==="idle")enable();else if(stage==="ready")record();};
off.onclick=()=>idle("Microphone is off.");mic.onchange=()=>idle("Input selected. Enable it to check the level.");
picker.onchange=()=>{const file=picker.files[0];picker.value="";choose(file);};
for(const name of ["dragenter","dragover"]){box.addEventListener(name,e=>{e.preventDefault();box.classList.add("drag");});}
box.addEventListener("dragleave",()=>box.classList.remove("drag"));box.addEventListener("drop",e=>{e.preventDefault();box.classList.remove("drag");choose(e.dataTransfer.files[0]);});
window.addEventListener("pagehide",()=>{permissionVersion++;if(recorder){recorder.onstop=null;if(recorder.state!=="inactive")recorder.stop();recorder=null;}release();if(url)URL.revokeObjectURL(url);});
</script></body></html>"""
