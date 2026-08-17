import React, { useState, useEffect } from 'react';

export default function AngiogramQCAView() {
  const [patientId, setPatientId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [file, setFile] = useState(null);
  const [rawPreview, setRawPreview] = useState(null);
  const [status, setStatus] = useState('idle'); // 'idle' | 'processing' | 'complete'
  const [qcaData, setQcaData] = useState(null);

  const [zoomScale, setZoomScale] = useState(1.0);
  const [geminiReport, setGeminiReport] = useState(null);
  const [isGeneratingGemini, setIsGeneratingGemini] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has('patient_id')) setPatientId(params.get('patient_id'));
    if (params.has('session_id')) setSessionId(params.get('session_id'));
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      if (selected.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (ev) => setRawPreview(ev.target.result);
        reader.readAsDataURL(selected);
      } else {
        setRawPreview(null);
      }
    }
  };

  const handleRunQCA = async () => {
    if (!file) {
      alert('Please select an angiogram file (DICOM, MP4, PNG, JPG) first.');
      return;
    }

    setStatus('processing');
    const formData = new FormData();
    formData.append('angio_file', file);
    formData.append('patient_id', patientId);
    formData.append('session_id', sessionId);

    try {
      const response = await fetch('/api/v1/angiogram/process_video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setQcaData(data);
      setStatus('complete');
    } catch (err) {
      alert('QCA Inspection failed: ' + err.message);
      setStatus('idle');
    }
  };

  const handleGenerateGeminiReport = async () => {
    setIsGeneratingGemini(true);
    const payload = {
      session_id: sessionId || null,
      deepsa_output: qcaData || { status: 'complete' },
    };

    try {
      const res = await fetch('/api/v1/reports/ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setGeminiReport(data);
    } catch (err) {
      alert('Gemini multi-modal report error: ' + err.message);
    } finally {
      setIsGeneratingGemini(false);
    }
  };

  const metrics = qcaData?.qca_metrics || {};
  const stenosisPct = metrics.stenosis_percentage || 0;
  const severity = metrics.severity_grade || (stenosisPct >= 70 ? 'SEVERE' : (stenosisPct >= 50 ? 'MODERATE' : 'MILD'));

  return (
    <div className="min-h-screen bg-[#06080D] text-slate-200 font-sans p-6">
      <header className="max-w-7xl mx-auto flex items-center justify-between border-b border-white/10 pb-4 mb-8">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00E676] to-[#00D2FF] flex items-center justify-center font-bold text-black">
            <i className="fa-solid fa-vial-circle-check"></i>
          </div>
          <span className="text-xl font-extrabold text-white tracking-wider uppercase">CARDIO<span className="text-[#00E676]">-AI</span> QCA</span>
        </div>
        <a href="/upload_ecg.html" className="text-xs font-semibold text-slate-400 hover:text-white">&larr; Back to 12-Lead ECG</a>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT CONTROL PANEL */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
            <h2 className="text-xs font-mono font-bold text-[#00E676] uppercase tracking-wider">Cine & Keyframe Ingestion Zone</h2>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Patient ID</label>
                <input type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)} placeholder="PAT-9912" className="w-full p-2.5 rounded-xl bg-[#06080D] border border-white/10 text-white font-mono" />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Session ID</label>
                <input type="text" value={sessionId} onChange={(e) => setSessionId(e.target.value)} placeholder="Optional" className="w-full p-2.5 rounded-xl bg-[#06080D] border border-white/10 text-white font-mono" />
              </div>
            </div>

            <div className="text-xs">
              <label className="block text-slate-400 mb-1">Upload File (DICOM, MP4, PNG, JPG)</label>
              <input type="file" accept=".dcm,.mp4,.png,.jpg,.jpeg" onChange={handleFileChange} className="w-full text-slate-300 text-xs" />
              {file && <p className="mt-1 text-[11px] text-[#00D2FF] font-mono">Selected: {file.name}</p>}
            </div>

            <button
              onClick={handleRunQCA}
              disabled={status === 'processing'}
              className="w-full py-3.5 rounded-full bg-gradient-to-r from-[#00E676] to-[#00D2FF] text-black font-extrabold text-xs uppercase tracking-wider hover:scale-105 transition-all shadow-[0_0_20px_rgba(0,230,118,0.4)]"
            >
              {status === 'processing' ? 'Executing DeepSA Pipeline...' : 'Execute DeepSA QCA Inspection'}
            </button>
          </div>

          {/* AI MULTI-MODAL SYNTHESIS CARD */}
          <div className="bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
            <h3 className="text-xs font-mono font-bold text-[#00D2FF] uppercase tracking-wider">AI Multi-Modal Synthesis Card</h3>
            <p className="text-xs text-slate-300">Synthesize multi-engine findings (CKM 99% risk + ECG RCA MI + DeepSA QCA) via Gemini 1.5 Pro.</p>
            <button
              onClick={handleGenerateGeminiReport}
              disabled={isGeneratingGemini}
              className="w-full py-3 rounded-full bg-[#00D2FF]/10 text-[#00D2FF] border border-[#00D2FF]/40 font-bold text-xs uppercase hover:bg-[#00D2FF] hover:text-black transition-all"
            >
              {isGeneratingGemini ? 'Generating Multi-Modal Report...' : 'Generate AI Multi-Modal Synthesis'}
            </button>

            {geminiReport && (
              <div className="bg-[#06080D] p-4 rounded-2xl border border-[#00D2FF]/40 space-y-3">
                <div>
                  <div className="text-[10px] font-mono uppercase text-[#00D2FF] font-bold">3-SENTENCE CLINICAL IMPRESSION</div>
                  <p className="text-xs text-slate-200 mt-1">{geminiReport.clinical_summary}</p>
                </div>
                <div className="border-t border-white/10 pt-2">
                  <div class="text-[10px] font-mono uppercase text-[#00E676] font-bold">PATIENT ACTION PLAN</div>
                  <p className="text-xs text-slate-300 mt-1 whitespace-pre-line">{geminiReport.patient_next_steps}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT DUAL-VIEW & METRICS PANEL */}
        <div className="lg:col-span-8 space-y-6">
          {/* DUAL VIEW CANVAS */}
          <div className="bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
            <div className="flex justify-between items-center text-xs font-mono text-[#00E676]">
              <span>QCA DUAL-VIEW DIAGNOSTIC VISUALIZER</span>
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-400">{status === 'complete' ? 'QCA COMPLETE' : 'STANDBY'}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* PANEL A: RAW FRAME WITH ZOOM */}
              <div className="bg-[#06080D] p-3 rounded-2xl border border-white/10 space-y-2 text-center">
                <div className="flex justify-between text-[11px] font-mono text-slate-400">
                  <span>PANEL A: RAW FRAME</span>
                  <div className="space-x-1">
                    <button onClick={() => setZoomScale(Math.min(3, zoomScale + 0.2))} className="px-2 py-0.5 rounded bg-white/10 text-white">+</button>
                    <button onClick={() => setZoomScale(Math.max(0.5, zoomScale - 0.2))} className="px-2 py-0.5 rounded bg-white/10 text-white">-</button>
                    <button onClick={() => setZoomScale(1.0)} className="px-2 py-0.5 rounded bg-white/10 text-white">Reset</button>
                  </div>
                </div>
                <div className="w-full h-64 bg-[#0A0D14] rounded-xl flex items-center justify-center overflow-hidden">
                  {rawPreview ? (
                    <img src={rawPreview} alt="Raw frame" style={{ transform: `scale(${zoomScale})` }} className="max-h-full object-contain transition-transform" />
                  ) : (
                    <p className="text-xs text-slate-500">Raw keyframe preview</p>
                  )}
                </div>
              </div>

              {/* PANEL B: ANNOTATED QCA VISUALIZATION */}
              <div className="bg-[#06080D] p-3 rounded-2xl border border-[#00E676]/30 space-y-2 text-center">
                <div className="flex justify-between text-[11px] font-mono text-[#00E676] font-bold">
                  <span>PANEL B: DeepSA ARTERIAL TREE</span>
                  <span>Annotated (d_min)</span>
                </div>
                <div className="w-full h-64 bg-[#0A0D14] rounded-xl flex items-center justify-center overflow-hidden">
                  {qcaData?.qca_image_url ? (
                    <img src={qcaData.qca_image_url} alt="DeepSA QCA visual report" className="max-h-full object-contain" />
                  ) : (
                    <div className="text-slate-500 text-xs space-y-1">
                      <i className="fa-solid fa-crosshairs text-2xl text-[#00E676]/40"></i>
                      <p>DeepSA Medial Axis & Crosshair report</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* QUANTITATIVE METRICS PANEL */}
          <div className="bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
            <h3 className="text-xs font-mono font-bold text-[#00D2FF] uppercase tracking-wider">Quantitative Stenosis Metrics & Severity</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
              <div className="bg-[#06080D] p-3 rounded-xl">
                <span className="text-slate-400 block text-[10px]">STENOSIS %</span>
                <span className="text-xl font-bold text-white">{Math.round(stenosisPct)}%</span>
              </div>
              <div className="bg-[#06080D] p-3 rounded-xl">
                <span className="text-slate-400 block text-[10px]">d_min (MIN LUMEN)</span>
                <span className="text-xl font-bold text-[#00D2FF]">{(metrics.d_min || 0).toFixed(2)} mm</span>
              </div>
              <div className="bg-[#06080D] p-3 rounded-xl">
                <span className="text-slate-400 block text-[10px]">d_ref (REF VESSEL)</span>
                <span className="text-xl font-bold text-white">{(metrics.d_ref || 0).toFixed(2)} mm</span>
              </div>
              <div className="bg-[#06080D] p-3 rounded-xl">
                <span className="text-slate-400 block text-[10px]">SEVERITY GRADE</span>
                <span className={`text-sm font-bold ${severity === 'SEVERE' ? 'text-[#FF3D57]' : (severity === 'MODERATE' ? 'text-[#FFB300]' : 'text-[#00E676]')}`}>{severity}</span>
              </div>
            </div>

            <div className={`p-4 rounded-2xl border ${severity === 'SEVERE' ? 'bg-[#FF3D57]/10 border-[#FF3D57]/60 text-[#FF3D57]' : 'bg-[#00E676]/10 border-[#00E676]/30 text-[#00E676]'} font-bold text-xs`}>
              Intervention Recommendation: {severity === 'SEVERE' ? 'Catheter Intervention / Stent Placement Indicated' : 'Conservative Medical Management'}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
