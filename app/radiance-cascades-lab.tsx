"use client";

import { useEffect } from "react";

export function RadianceCascadesLab() {
  useEffect(() => {
    const script = document.createElement("script");
    script.type = "module";
    script.src = "/rc/engine.js";
    script.dataset.splitRcLoader = "true";
    document.head.append(script);
    return () => {
      script.remove();
      window.__splitRC?.destroy?.();
    };
  }, []);

  return (
    <main id="app-shell" data-testid="split-rc-app">
      <canvas id="viewport" aria-label="Split Radiance Cascades WebGPU viewport" />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Split RC</h1>
            <p>world-space WebGPU GI lab</p>
          </div>
        </div>
        <div className="metrics" aria-live="polite">
          <span><b id="metric-fps">—</b><small>FPS</small></span>
          <span><b id="metric-gpu">—</b><small>GPU MS</small></span>
          <span><b id="metric-rays">—</b><small>RAYS</small></span>
          <span><b id="metric-probes">—</b><small>PROBES</small></span>
        </div>
        <button id="toggle-panel" className="icon-button" aria-label="Toggle controls">Controls</button>
      </header>

      <aside id="control-panel" className="control-panel" aria-label="Renderer controls">
        <section>
          <p className="eyebrow">Validation scene</p>
          <div className="scene-heading">
            <div>
              <h2 id="scene-name">Color bleed laboratory</h2>
              <p id="scene-description">Loading production renderer…</p>
            </div>
            <span id="scene-index">01 / 10</span>
          </div>
          <div className="scene-nav">
            <button id="prev-scene" aria-label="Previous test scene">←</button>
            <select id="scene-select" aria-label="Choose test scene" />
            <button id="next-scene" aria-label="Next test scene">→</button>
          </div>
        </section>

        <section className="control-grid">
          <label>
            <span>View</span>
            <select id="debug-view">
              <option value="0">Final composite</option>
              <option value="1">Indirect only</option>
              <option value="2">Direct only</option>
              <option value="3">Surface normals</option>
              <option value="4">Probe coverage</option>
              <option value="5">Albedo</option>
            </select>
          </label>
          <label>
            <span>Quality</span>
            <select id="quality" defaultValue="balanced">
              <option value="performance">Performance</option>
              <option value="balanced">Balanced</option>
              <option value="quality">Quality</option>
              <option value="ultra">Ultra</option>
            </select>
          </label>
        </section>

        <section>
          <label className="range-label" htmlFor="indirect-strength">
            <span>Indirect strength</span><output id="indirect-value">1.00×</output>
          </label>
          <input id="indirect-strength" type="range" min="0" max="2" step="0.05" defaultValue="1" />

          <label className="range-label" htmlFor="sun-speed">
            <span>Sun animation</span><output id="sun-value">1.00×</output>
          </label>
          <input id="sun-speed" type="range" min="0" max="2" step="0.05" defaultValue="1" />
        </section>

        <section className="toggles">
          <label><input id="animate-camera" type="checkbox" defaultChecked /><span>Camera path</span></label>
          <label><input id="animate-lights" type="checkbox" defaultChecked /><span>Moving lights</span></label>
          <label><input id="temporal-stability" type="checkbox" defaultChecked /><span>Stable history</span></label>
          <label><input id="show-profiler" type="checkbox" defaultChecked /><span>Pass profiler</span></label>
        </section>

        <section id="pass-profiler" className="profiler">
          <div><span>GI grid</span><i><b style={{ width: "100%" }} /></i><output id="gi-resolution">—</output></div>
          <div><span>Frame</span><i><b id="bar-frame" /></i><output id="pass-frame">—</output></div>
          <div><span>Geometry</span><i><b id="bar-geometry" /></i><output id="pass-geometry">—</output></div>
          <div><span>Ray split + merge</span><i><b id="bar-gi" /></i><output id="pass-gi">—</output></div>
          <div><span>Composite</span><i><b id="bar-composite" /></i><output id="pass-composite">—</output></div>
        </section>

        <footer>
          <span id="gpu-name">Detecting GPU…</span>
          <button id="run-validation">Run 10-scene audit</button>
        </footer>
      </aside>

      <div className="scene-strip" id="scene-strip" aria-label="Test scene shortcuts" />

      <div className="status-card" id="status-card" role="status">
        <span className="spinner" aria-hidden="true" />
        <div><b id="status-title">Initializing WebGPU</b><p id="status-detail">Requesting a high-performance adapter…</p></div>
      </div>

      <div className="audit-card" id="audit-card" hidden>
        <div className="audit-heading">
          <div><p className="eyebrow">Automated audit</p><h2 id="audit-title">Running scene 1 of 10</h2></div>
          <button id="close-audit" aria-label="Close audit report">×</button>
        </div>
        <div className="audit-progress"><i id="audit-progress" /></div>
        <pre id="audit-report">Warming renderer…</pre>
      </div>

      <div className="help">
        <span><kbd>Drag</kbd> orbit</span><span><kbd>Wheel</kbd> zoom</span><span><kbd>WASD</kbd> move</span><span><kbd>Space</kbd> pause</span>
      </div>
    </main>
  );
}

declare global {
  interface Window {
    __splitRC?: { destroy?: () => void };
  }
}
