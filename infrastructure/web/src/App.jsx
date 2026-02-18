import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Play, Pause, RefreshCw, CheckCircle,
    User, Image as ImageIcon, MessageSquare,
    ChevronRight, ChevronDown, Layers,
    Folder, Calendar, Clock, ArrowLeft, LayoutGrid
} from 'lucide-react';

function App() {
    const [state, setState] = useState(null);
    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(true);
    const [view, setView] = useState('dashboard'); // 'dashboard' | 'detail'
    const [activeScene, setActiveScene] = useState(null);
    const [inputHash, setInputHash] = useState(null);

    useEffect(() => {
        fetchProjects();
        if (inputHash) {
            fetchState();
        }
    }, [inputHash]);

    const fetchProjects = async () => {
        try {
            const res = await axios.get('/api/projects');
            setProjects(res.data);
        } catch (err) {
            console.error("Failed to fetch projects:", err);
        }
    };

    const fetchState = async () => {
        if (!inputHash) return;
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

    const handleProjectSelect = (hash) => {
        setInputHash(hash);
        setView('detail');
    };

    if (loading) return <div className="loading">Initializing Dashboard...</div>;

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="sidebar glass">
                <div className="sidebar-header" onClick={() => setView('dashboard')} style={{ cursor: 'pointer' }}>
                    <Layers className="text-gradient" />
                    <h2 className="text-gradient">ComicGen Pro</h2>
                </div>

                <nav className="nav-menu">
                    <div
                        className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}
                        onClick={() => setView('dashboard')}
                    >
                        <LayoutGrid size={18} />
                        <span>All Projects</span>
                    </div>
                </nav>

                {view === 'detail' && state && (
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
                )}
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {view === 'dashboard' ? (
                    <div className="dashboard-view">
                        <header className="page-header">
                            <h1>Project Dashboard</h1>
                            <p>Manage and monitor all your ongoing comic generations.</p>
                        </header>

                        <div className="projects-grid">
                            {projects.map(project => (
                                <div key={project.input_hash} className="project-card glass" onClick={() => handleProjectSelect(project.input_hash)}>
                                    <div className="project-icon">
                                        <Folder size={32} />
                                    </div>
                                    <div className="project-meta">
                                        <div className="status-tag">{project.stage}</div>
                                        <h3>{project.name}</h3>
                                        <div className="project-stats">
                                            <span><strong>{project.scenes_total}</strong> Scenes</span>
                                            <span><strong>{project.pages_generated}</strong> Pages</span>
                                        </div>
                                    </div>
                                    <div className="project-footer">
                                        <div className="timestamp">
                                            <Clock size={12} />
                                            {new Date(project.timestamp * 1000).toLocaleDateString()}
                                        </div>
                                        <ChevronRight size={18} />
                                    </div>
                                </div>
                            ))}
                            <div className="project-card glass add-new dashed">
                                <Play size={32} />
                                <h3>New Generation</h3>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="detail-view">
                        <header className="main-header glass">
                            <div className="header-left">
                                <button className="back-btn" onClick={() => setView('dashboard')}>
                                    <ArrowLeft size={18} />
                                </button>
                                <div className="status-badge">
                                    <span className="dot pulse"></span>
                                    LIVE: {state?.stage || 'Initializing'}
                                </div>
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
                    </div>
                )}
            </main>

            <style dangerouslySetInnerHTML={{
                __html: `
        .app-container { display: flex; height: 100vh; overflow: hidden; background: #0d1117; color: #fff; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; }
        .text-gradient { background: linear-gradient(135deg, #60efff 0%, #00ff87 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        .sidebar { width: 260px; padding: 1.5rem; border-right: 1px solid rgba(255,255,255,0.1); }
        .sidebar-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2.5rem; }
        
        .nav-menu { margin-bottom: 2rem; }
        .nav-item { display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem; border-radius: 8px; cursor: pointer; transition: background 0.3s; opacity: 0.7; }
        .nav-item:hover { background: rgba(255,255,255,0.05); opacity: 1; }
        .nav-item.active { background: rgba(96, 239, 255, 0.1); color: #60efff; opacity: 1; }

        .scene-list h3 { font-size: 0.75rem; text-transform: uppercase; opacity: 0.4; letter-spacing: 0.1em; margin-bottom: 1rem; margin-top: 2rem; }
        .scene-item { display: flex; justify-content: space-between; align-items: center; padding: 0.7rem 1rem; border-radius: 8px; cursor: pointer; margin-bottom: 0.4rem; transition: all 0.2s; font-size: 0.9rem; }
        .scene-item:hover { background: rgba(255,255,255,0.05); }
        .scene-item.active { background: rgba(96, 239, 255, 0.1); color: #60efff; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }

        .main-content { flex: 1; padding: 2rem; overflow-y: auto; background: radial-gradient(circle at top right, rgba(96, 239, 255, 0.05), transparent); }
        
        .page-header { margin-bottom: 2.5rem; }
        .page-header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .page-header p { opacity: 0.6; }

        .projects-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }
        .project-card { padding: 1.5rem; transition: transform 0.3s, box-shadow 0.3s; cursor: pointer; position: relative; overflow: hidden; }
        .project-card:hover { transform: translateY(-5px); box-shadow: 0 12px 24px rgba(0,0,0,0.3); border-color: rgba(96, 239, 255, 0.3); }
        .project-card.dashed { border-style: dashed; display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.5; }
        .project-card.dashed:hover { opacity: 1; border-color: #60efff; }

        .project-icon { width: 56px; height: 56px; background: rgba(96, 239, 255, 0.1); color: #60efff; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.2rem; }
        .status-tag { position: absolute; top: 1.5rem; right: 1.5rem; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; background: rgba(255,255,255,0.05); padding: 0.3rem 0.6rem; border-radius: 20px; color: #60efff; border: 1px solid rgba(96, 239, 255, 0.2); }
        .project-meta h3 { margin: 0 0 1rem; font-size: 1.1rem; }
        .project-stats { display: flex; gap: 1.5rem; font-size: 0.85rem; opacity: 0.7; }
        .project-stats strong { color: #fff; }
        
        .project-footer { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center; }
        .timestamp { font-size: 0.75rem; opacity: 0.5; display: flex; align-items: center; gap: 0.4rem; }

        .main-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
        .header-left { display: flex; align-items: center; gap: 1rem; }
        .back-btn { background: none; border: none; color: #fff; cursor: pointer; opacity: 0.6; transition: opacity 0.2s; padding: 0.5rem; }
        .back-btn:hover { opacity: 1; }

        .status-badge { display: flex; align-items: center; gap: 0.6rem; font-weight: 600; font-size: 0.9rem; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #00ff87; }
        .pulse { animation: pulse-animation 2s infinite; }
        @keyframes pulse-animation { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

        .global-controls { display: flex; gap: 1rem; }
        .primary-btn { background: #60efff; color: #000; display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600; cursor: pointer; }
        .secondary-btn { background: rgba(255,255,255,0.05); color: #fff; display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; border: 1px solid rgba(255,255,255,0.1); }

        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr 300px; gap: 1.5rem; height: calc(100vh - 160px); }
        .card { padding: 1.5rem; display: flex; flex-direction: column; }
        .card h3 { margin: 0 0 1.2rem; display: flex; align-items: center; gap: 0.6rem; font-size: 1rem; font-weight: 600; }
        .script-editor { flex: 1; overflow: hidden; }
        .script-editor pre { white-space: pre-wrap; font-size: 0.85rem; height: 100%; overflow: auto; background: rgba(0,0,0,0.3); padding: 1.2rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }
        
        .panels-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-top: 1rem; }
        .panel-card { padding: 1.2rem; border-radius: 12px; }
        .panel-placeholder { height: 180px; background: rgba(0,0,0,0.3); border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
        .panel-placeholder img { width: 100%; height: 100%; object-fit: cover; }
        .panel-info h4 { margin: 0 0 0.5rem; font-size: 1rem; }
        .panel-info p { font-size: 0.8rem; opacity: 0.7; margin-bottom: 1.2rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .action-btn { background: rgba(96, 239, 255, 0.1); border: 1px solid rgba(96, 239, 255, 0.2); color: #60efff; display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
        
        .char-list { flex: 1; overflow-y: auto; }
        .char-item { padding: 1rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.05); }
        .char-item strong { display: block; margin-bottom: 0.4rem; color: #60efff; }
        .char-item p { font-size: 0.8rem; opacity: 0.7; margin: 0; line-height: 1.4; }
        .loading { display: flex; align-items: center; justify-content: center; height: 100vh; font-size: 1.2rem; font-weight: 600; color: #60efff; letter-spacing: 0.1em; }
      `}} />
        </div>
    );
}

export default App;
