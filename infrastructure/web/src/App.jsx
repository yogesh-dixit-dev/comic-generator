import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Play, Pause, RefreshCw, CheckCircle,
    User, Image as ImageIcon, MessageSquare,
    ChevronRight, ChevronDown, Layers
} from 'lucide-react';

function App() {
    const [state, setState] = useState(null);
    const [loading, setLoading] = useState(true);
    const [activeScene, setActiveScene] = useState(null);
    const [inputHash, setInputHash] = useState('abc'); // Mock for now

    useEffect(() => {
        fetchState();
    }, [inputHash]);

    const fetchState = async () => {
        try {
            setLoading(true);
            const res = await axios.get(`/api/state/${inputHash}`);
            setState(res.data);
            if (res.data?.master_script?.scenes?.length > 0 && !activeScene) {
                setActiveScene(res.data.master_script.scenes[0].id);
            }
        } catch (err) {
            console.error("Failed to fetch state:", err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="loading">Initializing Dashboard...</div>;

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="sidebar glass">
                <div className="sidebar-header">
                    <Layers className="text-gradient" />
                    <h2 className="text-gradient">ComicGen Pro</h2>
                </div>

                <nav className="scene-list">
                    <h3>Scenes</h3>
                    {state?.master_script?.scenes?.map(scene => (
                        <div
                            key={scene.id}
                            className={`scene-item ${activeScene === scene.id ? 'active' : ''}`}
                            onClick={() => setActiveScene(scene.id)}
                        >
                            <span>{scene.label || `Scene ${scene.id}`}</span>
                            {activeScene === scene.id && <ChevronRight size={16} />}
                        </div>
                    ))}
                </nav>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                <header className="main-header glass">
                    <div className="status-badge">
                        <span className="dot pulse"></span>
                        LIVE: {state?.stage || 'Initializing'}
                    </div>
                    <div className="global-controls">
                        <button className="primary-btn"><Play size={18} /> Resume Pipeline</button>
                        <button className="secondary-btn"><Pause size={18} /> Global Pause</button>
                    </div>
                </header>

                <section className="dashboard-grid">
                    {/* Script Section */}
                    <div className="card glass">
                        <h3><MessageSquare size={20} /> Script Context</h3>
                        <div className="script-editor">
                            <pre contentEditable>{JSON.stringify(state?.master_script?.scenes.find(s => s.id === activeScene), null, 2)}</pre>
                        </div>
                    </div>

                    {/* Panels Preview */}
                    <div className="card glass panels-view">
                        <h3><ImageIcon size={20} /> Panels (Scene {activeScene})</h3>
                        <div className="panels-grid">
                            {state?.master_script?.scenes.find(s => s.id === activeScene)?.panels?.map(panel => (
                                <div key={panel.id} className="panel-card glass">
                                    <div className="panel-placeholder">
                                        {panel.image_path ? <img src={`/output/${panel.image_path}`} alt="Panel" /> : <ImageIcon size={48} opacity={0.3} />}
                                    </div>
                                    <div className="panel-info">
                                        <h4>Panel {panel.id}</h4>
                                        <p>{panel.description}</p>
                                        <button className="action-btn"><RefreshCw size={14} /> Regenerate</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Characters Sidebar */}
                    <div className="card glass characters-view">
                        <h3><User size={20} /> Characters</h3>
                        <div className="char-list">
                            {state?.characters?.map(char => (
                                <div key={char.name} className="char-item glass">
                                    <strong>{char.name}</strong>
                                    <p>{char.description}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            </main>

            <style dangerouslySetInnerHTML={{
                __html: `
        .app-container { display: flex; height: 100vh; overflow: hidden; background: #0d1117; }
        .sidebar { width: 260px; padding: 1.5rem; border-right: 1px solid rgba(255,255,255,0.1); }
        .sidebar-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2rem; }
        .sidebar-header h2 { margin: 0; font-size: 1.2rem; }
        
        .scene-list h3 { font-size: 0.8rem; text-transform: uppercase; opacity: 0.5; margin-bottom: 1rem; }
        .scene-item { display: flex; justify-content: space-between; align-items: center; padding: 0.8rem; border-radius: 8px; cursor: pointer; margin-bottom: 0.5rem; transition: background 0.2s; }
        .scene-item:hover { background: rgba(255,255,255,0.05); }
        .scene-item.active { background: rgba(96, 239, 255, 0.1); color: #60efff; }

        .main-content { flex: 1; padding: 1.5rem; overflow-y: auto; }
        .main-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
        .status-badge { display: flex; align-items: center; gap: 0.6rem; font-weight: 600; font-size: 0.9rem; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #00ff87; }
        .pulse { animation: pulse-animation 2s infinite; }
        @keyframes pulse-animation { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

        .global-controls { display: flex; gap: 1rem; }
        .primary-btn { background: #60efff; color: #000; display: flex; align-items: center; gap: 0.5rem; }
        .secondary-btn { background: rgba(255,255,255,0.05); color: #fff; display: flex; align-items: center; gap: 0.5rem; }

        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr 300px; gap: 1.5rem; height: calc(100vh - 120px); }
        .script-editor pre { white-space: pre-wrap; font-size: 0.85rem; height: 100%; overflow: auto; background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px; }
        
        .panels-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
        .panel-card { padding: 1rem; border-radius: 12px; }
        .panel-placeholder { height: 150px; background: rgba(0,0,0,0.3); border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 0.8rem; overflow: hidden; }
        .panel-placeholder img { width: 100%; height: 100%; object-fit: cover; }
        .panel-info h4 { margin: 0 0 0.5rem; font-size: 1rem; }
        .panel-info p { font-size: 0.8rem; opacity: 0.7; margin-bottom: 1rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        
        .char-item { padding: 1rem; margin-bottom: 1rem; }
        .char-item strong { display: block; margin-bottom: 0.4rem; color: #60efff; }
        .char-item p { font-size: 0.8rem; opacity: 0.7; margin: 0; }
        .loading { display: flex; align-items: center; justify-content: center; height: 100vh; font-size: 1.2rem; font-weight: 600; color: #60efff; }
      `}} />
        </div>
    );
}

export default App;
