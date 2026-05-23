import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
    Play, Pause, RefreshCw, CheckCircle,
    User, Image as ImageIcon, MessageSquare,
    ChevronRight, ChevronDown, Layers,
    Folder, Calendar, Clock, ArrowLeft, LayoutGrid,
    Upload, Settings, Eye, Sliders, Shield, BookOpen, Plus
} from 'lucide-react';

function App() {
    const [state, setState] = useState(null);
    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(false);
    const [view, setView] = useState('dashboard'); // 'dashboard' | 'create' | 'detail'
    const [activeScene, setActiveScene] = useState(null);
    const [inputHash, setInputHash] = useState(null);

    // Form inputs for creating a new project
    const [projectName, setProjectName] = useState('');
    const [storyText, setStoryText] = useState('');
    const [stylePreset, setStylePreset] = useState('cinematic, detailed, comic book style');
    const [autoRun, setAutoRun] = useState(true);

    // Editing buffers
    const [editScriptData, setEditScriptData] = useState(null);
    const [editCharacters, setEditCharacters] = useState([]);
    const [editScenePlans, setEditScenePlans] = useState({});
    const [saveStatus, setSaveStatus] = useState('');

    const pollingRef = useRef(null);

    useEffect(() => {
        fetchProjects();
        return () => {
            if (pollingRef.current) clearInterval(pollingRef.current);
        };
    }, []);

    useEffect(() => {
        if (inputHash) {
            fetchState();
            // Start polling when in detail view
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = setInterval(fetchState, 3000);
        } else {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
            }
        }
        return () => {
            if (pollingRef.current) clearInterval(pollingRef.current);
        };
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
            const res = await axios.get(`/api/state/${inputHash}`);
            if (res.data?.error) {
                 console.error("State API returned error:", res.data.error);
                 return;
            }
            setState(res.data);
            
            // Set active scene if not set yet
            if (res.data?.master_script?.scenes?.length > 0 && !activeScene) {
                setActiveScene(res.data.master_script.scenes[0].id);
            }
            
            // Sync local editing buffers if not currently editing/dirty
            if (res.data?.master_script && !editScriptData) {
                setEditScriptData(JSON.parse(JSON.stringify(res.data.master_script)));
            }
            if (res.data?.characters && editCharacters.length === 0) {
                setEditCharacters(JSON.parse(JSON.stringify(res.data.characters)));
            }
            if (res.data?.scene_plans && Object.keys(editScenePlans).length === 0) {
                setEditScenePlans(JSON.parse(JSON.stringify(res.data.scene_plans)));
            }
        } catch (err) {
            console.error("Failed to fetch state:", err);
        }
    };

    const handleProjectSelect = (hash) => {
        // Clear editing buffers
        setEditScriptData(null);
        setEditCharacters([]);
        setEditScenePlans({});
        setActiveScene(null);
        setInputHash(hash);
        setView('detail');
    };

    const handleCreateProject = async (e) => {
        e.preventDefault();
        if (!storyText.trim()) {
            alert("Please paste or upload some story text.");
            return;
        }
        try {
            setLoading(true);
            const payload = {
                story_text: storyText,
                project_name: projectName || "Unnamed Project",
                style_preset: stylePreset,
                auto_run: autoRun
            };
            const res = await axios.post('/api/pipeline/start', payload);
            fetchProjects();
            
            // Reset form
            setProjectName('');
            setStoryText('');
            
            // Open project
            handleProjectSelect(res.data.input_hash);
        } catch (err) {
            console.error("Failed to start pipeline:", err);
            alert("Error launching pipeline: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    const toggleAutoRun = async () => {
        if (!inputHash || !state) return;
        const newAutoRun = !state.metadata.auto_run;
        try {
            await axios.post(`/api/state/${inputHash}/toggle-auto-run`, { auto_run: newAutoRun });
            fetchState();
        } catch (err) {
            console.error("Failed to toggle auto_run:", err);
        }
    };

    const stopPipeline = async () => {
        if (!inputHash) return;
        try {
            await axios.post(`/api/pipeline/stop/${inputHash}`);
            fetchState();
        } catch (err) {
            console.error("Failed to stop pipeline:", err);
        }
    };

    const handleScriptChange = (sceneId, panelId, field, val) => {
        if (!editScriptData) return;
        const copy = JSON.parse(JSON.stringify(editScriptData));
        const scene = copy.scenes.find(s => s.id === sceneId);
        if (scene) {
            const panel = scene.panels.find(p => p.id === panelId);
            if (panel) {
                panel[field] = val;
                setEditScriptData(copy);
            }
        }
    };

    const handleSceneMetadataChange = (sceneId, field, val) => {
        if (!editScriptData) return;
        const copy = JSON.parse(JSON.stringify(editScriptData));
        const scene = copy.scenes.find(s => s.id === sceneId);
        if (scene) {
            scene[field] = val;
            setEditScriptData(copy);
        }
    };

    const handleCharacterChange = (charIndex, field, val) => {
        const copy = [...editCharacters];
        copy[charIndex][field] = val;
        setEditCharacters(copy);
    };

    const handleScenePlanChange = (sceneId, panelId, field, val) => {
        const copy = { ...editScenePlans };
        // Resolve sceneId as int or string to match data structures
        let plan = copy[sceneId] || copy[String(sceneId)];
        if (plan) {
            const panel = plan.panels.find(p => p.id === panelId);
            if (panel) {
                panel[field] = val;
                setEditScenePlans(copy);
            }
        }
    };

    const saveStateEdits = async () => {
        if (!inputHash) return;
        try {
            setSaveStatus('Saving changes...');
            const payload = {
                master_script: editScriptData,
                characters: editCharacters,
                scene_plans: editScenePlans
            };
            await axios.post(`/api/state/${inputHash}/update`, payload);
            setSaveStatus('✅ Changes saved to disk successfully!');
            setTimeout(() => setSaveStatus(''), 4000);
            fetchState();
        } catch (err) {
            console.error("Failed to save changes:", err);
            setSaveStatus('❌ Error saving changes: ' + err.message);
        }
    };

    const approveStage = async (stage) => {
        if (!inputHash) return;
        try {
            setSaveStatus(`Approving and resuming stage: ${stage}...`);
            const payload = {
                master_script: editScriptData,
                characters: editCharacters,
                scene_plans: editScenePlans
            };
            await axios.post(`/api/state/${inputHash}/approve/${stage}`, payload);
            setSaveStatus(`✅ Stage '${stage}' approved! Construction resumed.`);
            setTimeout(() => setSaveStatus(''), 4000);
            
            // Clear local editing dirty flags to pull updated downstream data
            setEditScriptData(null);
            setEditCharacters([]);
            setEditScenePlans([]);
            
            fetchState();
        } catch (err) {
            console.error("Failed to approve stage:", err);
            setSaveStatus('❌ Approval error: ' + err.message);
        }
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            setStoryText(evt.target.result);
        };
        reader.readAsText(file);
    };

    // Calculate progression metrics
    const getActiveProgressIndex = (stage) => {
        if (stage === "scripting") return 0;
        if (stage === "design") return 1;
        if (state?.metadata?.current_step === "planning") return 2;
        if (stage === "production" && state?.last_scene_id !== -1) return 3;
        return 4; // Complete / lettering
    };

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="sidebar glass">
                <div className="sidebar-header" onClick={() => { setView('dashboard'); setInputHash(null); }} style={{ cursor: 'pointer' }}>
                    <Layers className="text-gradient logo-icon" />
                    <h2 className="text-gradient">ComicGen HITL</h2>
                </div>

                <nav className="nav-menu">
                    <div
                        className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}
                        onClick={() => { setView('dashboard'); setInputHash(null); }}
                    >
                        <LayoutGrid size={18} />
                        <span>All Projects</span>
                    </div>
                    <div
                        className={`nav-item ${view === 'create' ? 'active' : ''}`}
                        onClick={() => { setView('create'); setInputHash(null); }}
                    >
                        <Plus size={18} />
                        <span>New Generation</span>
                    </div>
                </nav>

                {view === 'detail' && state && (
                    <nav className="scene-list">
                        <h3>Story Scenes</h3>
                        {state?.master_script?.scenes?.map(scene => (
                            <div
                                key={scene.id}
                                className={`scene-item ${activeScene === scene.id ? 'active' : ''}`}
                                onClick={() => setActiveScene(scene.id)}
                            >
                                <span className="scene-label">{scene.location ? scene.location.substring(0, 18) + (scene.location.length > 18 ? '...' : '') : `Scene ${scene.id}`}</span>
                                {activeScene === scene.id && <ChevronRight size={16} />}
                            </div>
                        ))}
                    </nav>
                )}
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {/* 1. DASHBOARD VIEW */}
                {view === 'dashboard' && (
                    <div className="dashboard-view animate-fade-in">
                        <header className="page-header">
                            <h1>Project Workspace</h1>
                            <p>Monitor, review, and construct your multi-agent comic generations.</p>
                        </header>

                        <div className="projects-grid">
                            {projects.map(project => (
                                <div key={project.input_hash} className="project-card glass" onClick={() => handleProjectSelect(project.input_hash)}>
                                    <div className="project-icon">
                                        <Folder size={32} />
                                    </div>
                                    <div className="project-meta">
                                        <div className={`status-tag ${project.is_running ? 'running-pulse' : ''}`}>
                                            {project.is_running ? 'Constructing' : project.stage}
                                        </div>
                                        <h3>{project.name}</h3>
                                        <div className="project-stats">
                                            <span><strong>{project.scenes_total}</strong> Scenes</span>
                                            <span><strong>{project.pages_generated}</strong> Panels Drawn</span>
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
                            
                            <div className="project-card glass add-new dashed" onClick={() => setView('create')}>
                                <Plus size={36} className="add-icon text-gradient" />
                                <h3>Create New Project</h3>
                                <p>Upload novel text to initialize agents</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* 2. CREATE PROJECT VIEW */}
                {view === 'create' && (
                    <div className="create-view glass animate-fade-in">
                        <header className="page-header">
                            <h1>Start New Generation</h1>
                            <p>Upload a novel, script, or short story to begin the multi-agent design pipeline.</p>
                        </header>

                        <form onSubmit={handleCreateProject} className="create-form">
                            <div className="form-group">
                                <label>Project Name</label>
                                <input
                                    type="text"
                                    placeholder="E.g., The Samurai's Flower"
                                    value={projectName}
                                    onChange={(e) => setProjectName(e.target.value)}
                                    required
                                />
                            </div>

                            <div className="form-group">
                                <label>Artistic Preset / Style Guide</label>
                                <select value={stylePreset} onChange={(e) => setStylePreset(e.target.value)}>
                                    <option value="cinematic, detailed, comic book style">Cinematic Detail (Modern Comic)</option>
                                    <option value="retro vintage manga style, black and white ink, highly detailed screentone">Vintage Manga (Black & White Screentone)</option>
                                    <option value="watercolor illustration, soft edges, whimsical fantasy painting">Watercolor Fantasy (Soft Painting)</option>
                                    <option value="bold line pop art, halftone dots, vibrant color block style">Retro Pop Art (Vibrant Halftone)</option>
                                    <option value="gritty noir comic art, heavy shadows, high contrast, graphic novel">Noir Graphic Novel (High Contrast)</option>
                                </select>
                            </div>

                            <div className="form-group">
                                <div className="toggle-container glass">
                                    <div className="toggle-text">
                                        <h4>Auto-Run Pipeline (Uninterrupted Mode)</h4>
                                        <p>If enabled, the pipeline runs start-to-finish without stops. Disable to manually pause, edit scripts, modify character descriptions, and select camera directions at each step.</p>
                                    </div>
                                    <label className="switch">
                                        <input
                                            type="checkbox"
                                            checked={autoRun}
                                            onChange={() => setAutoRun(!autoRun)}
                                        />
                                        <span className="slider round"></span>
                                    </label>
                                </div>
                            </div>

                            <div className="form-group">
                                <div className="textarea-header">
                                    <label>Novel / Story Text</label>
                                    <label className="file-upload-btn">
                                        <Upload size={14} /> Upload .txt File
                                        <input type="file" accept=".txt" onChange={handleFileUpload} style={{ display: 'none' }} />
                                    </label>
                                </div>
                                <textarea
                                    rows={10}
                                    placeholder="Paste your story text or novel chunk here... E.g., In a neon-lit Tokyo slums, a cybernetic samurai discovers a biological cat..."
                                    value={storyText}
                                    onChange={(e) => setStoryText(e.target.value)}
                                    required
                                />
                            </div>

                            <button type="submit" className="primary-btn submit-btn" disabled={loading}>
                                {loading ? (
                                    <>
                                        <RefreshCw className="spinner" size={18} /> Starting Agents...
                                    </>
                                ) : (
                                    <>
                                        <Play size={18} /> Launch Construction Pipeline
                                    </>
                                )}
                            </button>
                        </form>
                    </div>
                )}

                {/* 3. DETAIL VIEW */}
                {view === 'detail' && state && (
                    <div className="detail-view animate-fade-in">
                        <header className="main-header glass">
                            <div className="header-left">
                                <button className="back-btn" onClick={() => { setView('dashboard'); setInputHash(null); }}>
                                    <ArrowLeft size={18} />
                                </button>
                                <div className="project-title-area">
                                    <h2>{state.metadata?.project_name || "Ongoing Project"}</h2>
                                    <div className="status-badge">
                                        <span className={`dot ${state.metadata?.is_running ? 'pulse green' : 'red'}`}></span>
                                        {state.metadata?.is_running ? `LIVE Stage: ${state.stage.toUpperCase()}` : "PIPELINE COMPLETED / PAUSED"}
                                    </div>
                                </div>
                            </div>

                            <div className="global-controls">
                                <div className="auto-run-status glass">
                                    <span>Continuous Process: <strong>{state.metadata?.auto_run ? 'ON' : 'OFF'}</strong></span>
                                    <button className={`toggle-btn-small ${state.metadata?.auto_run ? 'active' : ''}`} onClick={toggleAutoRun}>
                                        {state.metadata?.auto_run ? 'Pause Step Gates' : 'Activate Auto-Run'}
                                    </button>
                                </div>
                                {state.metadata?.is_running && (
                                    <button className="stop-btn" onClick={stopPipeline}>
                                        <Pause size={16} /> Stop Pipeline
                                    </button>
                                )}
                            </div>
                        </header>

                        {/* Pipeline Progress Stepper */}
                        <div className="stepper-card glass">
                            <h3>Pipeline Progress</h3>
                            <div className="stepper">
                                {[
                                    { label: "Scripting", desc: "Text compilation" },
                                    { label: "Character Design", desc: "Consistency sheets" },
                                    { label: "Visual Planning", desc: "Directorial setup" },
                                    { label: "Drawing", desc: "Stable Diffusion XL" },
                                    { label: "Lettering & PDF", desc: "Final Packaging" }
                                ].map((step, idx) => {
                                    const progressIdx = getActiveProgressIndex(state.stage);
                                    let status = "pending";
                                    if (idx < progressIdx) status = "completed";
                                    else if (idx === progressIdx) status = "active";
                                    
                                    return (
                                        <div key={idx} className={`step-node ${status}`}>
                                            <div className="step-circle">
                                                {status === "completed" ? <CheckCircle size={16} /> : <span>{idx + 1}</span>}
                                            </div>
                                            <div className="step-label">
                                                <h4>{step.label}</h4>
                                                <p>{step.desc}</p>
                                            </div>
                                            {idx < 4 && <div className="step-line"></div>}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Paused approval alert */}
                        {!state.metadata?.auto_run && state.metadata?.current_step && !state.metadata?.[`approved_${state.metadata.current_step}`] && (
                            <div className="alert-box glass pulse-border">
                                <Shield className="alert-icon text-gradient" size={24} />
                                <div className="alert-text">
                                    <h3>Review Requested: {state.metadata.current_step.toUpperCase()} Gate</h3>
                                    <p>The construction pipeline is paused here. Verify the generated results below, make any custom edits, and click the **Approve & Resume** button to carry changes forward.</p>
                                </div>
                                <button className="approve-action-btn" onClick={() => approveStage(state.metadata.current_step)}>
                                     Approve & Resume Stage
                                </button>
                            </div>
                        )}

                        {saveStatus && <div className="save-status-bar glass">{saveStatus}</div>}

                        {/* Main Interaction Split View */}
                        <section className="dashboard-grid">
                            
                            {/* TAB 1: Script Editor */}
                            <div className="card glass script-section">
                                <div className="card-header">
                                    <h3><MessageSquare size={18} /> Dynamic Script Context</h3>
                                    {editScriptData && (
                                        <button className="save-btn" onClick={saveStateEdits}>Save Script Edits</button>
                                    )}
                                </div>

                                {editScriptData?.scenes?.length > 0 ? (
                                    <div className="script-container">
                                        {/* Scene Info */}
                                        {(() => {
                                            const activeSceneObj = editScriptData.scenes.find(s => s.id === activeScene);
                                            if (!activeSceneObj) return <div className="no-data">Select a scene from the left panel to review.</div>;
                                            
                                            return (
                                                <div className="active-scene-editor animate-fade-in">
                                                    <div className="form-group-row">
                                                        <div className="form-group-sub">
                                                            <label>Scene Location Setting</label>
                                                            <input
                                                                type="text"
                                                                value={activeSceneObj.location}
                                                                onChange={(e) => handleSceneMetadataChange(activeScene, 'location', e.target.value)}
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="form-group-sub" style={{ marginBottom: '1.2rem' }}>
                                                        <label>Plot / Narrative Summary</label>
                                                        <textarea
                                                            rows={2}
                                                            value={activeSceneObj.narrative_summary}
                                                            onChange={(e) => handleSceneMetadataChange(activeScene, 'narrative_summary', e.target.value)}
                                                        />
                                                    </div>

                                                    <div className="panels-list-header">
                                                        <h4>Scene Panels & Dialogue</h4>
                                                    </div>

                                                    {activeSceneObj.panels?.map(panel => (
                                                        <div key={panel.id} className="panel-editor-card glass">
                                                            <div className="panel-num-tag">Panel {panel.id}</div>
                                                            <div className="form-group-sub" style={{ marginBottom: '0.8rem' }}>
                                                                <label>Visual Action Description</label>
                                                                <textarea
                                                                    rows={2}
                                                                    value={panel.description}
                                                                    onChange={(e) => handleScriptChange(activeScene, panel.id, 'description', e.target.value)}
                                                                />
                                                            </div>

                                                            {/* Dialogue Items */}
                                                            <div className="dialogue-items">
                                                                <label>Speakers & Dialogue Lines</label>
                                                                {panel.dialogue && panel.dialogue.length > 0 ? (
                                                                    panel.dialogue.map((dlg, dIdx) => (
                                                                        <div key={dIdx} className="dialogue-row">
                                                                            <input
                                                                                type="text"
                                                                                className="dlg-speaker"
                                                                                placeholder="Speaker"
                                                                                value={dlg.speaker}
                                                                                onChange={(e) => {
                                                                                    const dlgCopy = [...panel.dialogue];
                                                                                    dlgCopy[dIdx].speaker = e.target.value;
                                                                                    handleScriptChange(activeScene, panel.id, 'dialogue', dlgCopy);
                                                                                }}
                                                                            />
                                                                            <input
                                                                                type="text"
                                                                                className="dlg-text"
                                                                                placeholder="Dialogue Text"
                                                                                value={dlg.text}
                                                                                onChange={(e) => {
                                                                                    const dlgCopy = [...panel.dialogue];
                                                                                    dlgCopy[dIdx].text = e.target.value;
                                                                                    handleScriptChange(activeScene, panel.id, 'dialogue', dlgCopy);
                                                                                }}
                                                                            />
                                                                        </div>
                                                                    ))
                                                                ) : (
                                                                    <p className="no-dialogue-label">No dialogue in this panel</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ))}
                                                    
                                                    {state.metadata?.current_step === "script" && (
                                                         <button className="approve-stage-btn" onClick={() => approveStage('script')}>
                                                                Approve & Complete Scripting
                                                         </button>
                                                    )}
                                                </div>
                                            );
                                        })()}
                                    </div>
                                ) : (
                                    <div className="no-data">
                                        <BookOpen size={48} opacity={0.3} style={{ marginBottom: '1rem' }} />
                                        <p>Scripting is active. Pipeline will generate scene details here in real-time.</p>
                                        <div className="loading-bar-placeholder pulse-bg"></div>
                                    </div>
                                )}
                            </div>

                            {/* TAB 2: Characters Inspector */}
                            <div className="card glass characters-section">
                                <div className="card-header">
                                    <h3><User size={18} /> Character Designs</h3>
                                    {editCharacters.length > 0 && (
                                        <button className="save-btn" onClick={saveStateEdits}>Save Character Edits</button>
                                    )}
                                </div>

                                {editCharacters.length > 0 ? (
                                    <div className="char-list-scroll">
                                        {editCharacters.map((char, cIdx) => (
                                            <div key={char.name} className="char-editor-card glass">
                                                <div className="char-editor-row">
                                                    <div className="form-group-sub flex-1">
                                                        <label>Canonical Name</label>
                                                        <input
                                                            type="text"
                                                            value={char.name}
                                                            onChange={(e) => handleCharacterChange(cIdx, 'name', e.target.value)}
                                                        />
                                                    </div>
                                                    <div className="form-group-sub flex-1">
                                                        <label>Pronouns</label>
                                                        <input
                                                            type="text"
                                                            value={char.pronouns}
                                                            onChange={(e) => handleCharacterChange(cIdx, 'pronouns', e.target.value)}
                                                        />
                                                    </div>
                                                </div>
                                                <div className="form-group-sub" style={{ marginTop: '0.8rem' }}>
                                                    <label>Visual Description (Physical Traits & Clothing)</label>
                                                    <textarea
                                                        rows={3}
                                                        value={char.description}
                                                        onChange={(e) => handleCharacterChange(cIdx, 'description', e.target.value)}
                                                    />
                                                </div>
                                                <div className="form-group-sub" style={{ marginTop: '0.8rem' }}>
                                                    <label>Personality Traits / Expression Notes</label>
                                                    <input
                                                        type="text"
                                                        value={char.personality}
                                                        onChange={(e) => handleCharacterChange(cIdx, 'personality', e.target.value)}
                                                    />
                                                </div>
                                            </div>
                                        ))}

                                        {state.metadata?.current_step === "characters" && (
                                             <button className="approve-stage-btn" onClick={() => approveStage('characters')}>
                                                    Approve Character Designs
                                             </button>
                                        )}
                                    </div>
                                ) : (
                                    <div className="no-data">
                                        <User size={48} opacity={0.3} style={{ marginBottom: '1rem' }} />
                                        <p>Character models are pending. The design agent will formulate individual character profiles here.</p>
                                        <div className="loading-bar-placeholder pulse-bg"></div>
                                    </div>
                                )}
                            </div>

                            {/* TAB 3: Visual Directing Plans */}
                            <div className="card glass visual-planning-section">
                                <div className="card-header">
                                    <h3><Sliders size={18} /> Visual & Camera Directions</h3>
                                    {Object.keys(editScenePlans).length > 0 && (
                                         <button className="save-btn" onClick={saveStateEdits}>Save Planning Edits</button>
                                    )}
                                </div>

                                {(() => {
                                    const activePlan = editScenePlans[activeScene] || editScenePlans[String(activeScene)];
                                    if (activePlan?.panels?.length > 0) {
                                        return (
                                            <div className="plan-list-scroll animate-fade-in">
                                                <p className="plan-helper-txt">Review and customize camera angles, lighting conditions, and prompts for each panel in **Scene {activeScene}** before starting SDXL drawing.</p>
                                                {activePlan.panels.map(panel => (
                                                    <div key={panel.id} className="plan-editor-card glass">
                                                        <div className="plan-panel-header">Panel {panel.id} Plan</div>
                                                        
                                                        <div className="form-group-row" style={{ display: 'flex', gap: '0.8rem', marginBottom: '0.8rem' }}>
                                                            <div className="form-group-sub flex-1">
                                                                <label>Camera Shot Angle</label>
                                                                <select 
                                                                    value={panel.camera_angle || 'Medium shot'}
                                                                    onChange={(e) => handleScenePlanChange(activeScene, panel.id, 'camera_angle', e.target.value)}
                                                                >
                                                                    <option value="Close-up shot">Close-up shot</option>
                                                                    <option value="Medium shot">Medium shot</option>
                                                                    <option value="Wide shot">Wide shot</option>
                                                                    <option value="Extreme wide shot">Extreme wide shot</option>
                                                                    <option value="Low-angle shot">Low-angle shot</option>
                                                                    <option value="High-angle shot">High-angle shot</option>
                                                                    <option value="Over-the-shoulder shot">Over-the-shoulder shot</option>
                                                                </select>
                                                            </div>
                                                            <div className="form-group-sub flex-1">
                                                                <label>Lighting Properties</label>
                                                                <select 
                                                                    value={panel.lighting || 'Dramatic shadows'}
                                                                    onChange={(e) => handleScenePlanChange(activeScene, panel.id, 'lighting', e.target.value)}
                                                                >
                                                                    <option value="Dramatic shadows">Dramatic shadows</option>
                                                                    <option value="Neon rim lighting">Neon rim lighting</option>
                                                                    <option value="Soft natural sunlight">Soft natural sunlight</option>
                                                                    <option value="Golden hour sunset glow">Golden hour sunset glow</option>
                                                                    <option value="Moody low-key noir lighting">Moody low-key noir</option>
                                                                    <option value="Vibrant cinematic studio lighting">Vibrant studio lighting</option>
                                                                </select>
                                                            </div>
                                                        </div>
                                                        
                                                        <div className="form-group-sub">
                                                            <label>Image Prompt Override</label>
                                                            <input 
                                                                type="text" 
                                                                value={panel.image_prompt || ''} 
                                                                onChange={(e) => handleScenePlanChange(activeScene, panel.id, 'image_prompt', e.target.value)}
                                                                placeholder="Override final Stable Diffusion prompt..."
                                                            />
                                                        </div>
                                                    </div>
                                                ))}

                                                {state.metadata?.current_step === "planning" && (
                                                     <button className="approve-stage-btn" onClick={() => approveStage('planning')}>
                                                            Approve Plans & Start Drawing
                                                     </button>
                                                )}
                                            </div>
                                        );
                                    }

                                    return (
                                        <div className="no-data">
                                             <Sliders size={48} opacity={0.3} style={{ marginBottom: '1rem' }} />
                                             <p>Visual planning plans will load here. The director agent will formulate scene camera shots, lighting values, and dynamic prompt structures in real-time.</p>
                                             <div className="loading-bar-placeholder pulse-bg"></div>
                                        </div>
                                    );
                                })()}
                            </div>

                        </section>

                        {/* Panels Drawing Preview Grid */}
                        <section className="panels-preview-box card glass animate-fade-in" style={{ marginTop: '1.5rem' }}>
                            <h3><ImageIcon size={20} /> Illustrated Panels Drawing Grid (Scene {activeScene})</h3>
                            {state?.master_script?.scenes.find(s => s.id === activeScene)?.panels?.length > 0 ? (
                                <div className="panels-grid">
                                    {state?.master_script?.scenes.find(s => s.id === activeScene)?.panels?.map(panel => (
                                        <div key={panel.id} className="panel-card glass">
                                            <div className="panel-placeholder">
                                                {panel.image_path ? (
                                                    <img src={`/output/${panel.image_path.replace(/\\/g, '/')}`} alt={`Scene ${activeScene} Panel ${panel.id}`} />
                                                ) : (
                                                    <div className="render-waiting">
                                                         <ImageIcon size={32} opacity={0.15} />
                                                         <p>Waiting for drawing phase...</p>
                                                    </div>
                                                )}
                                            </div>
                                            <div className="panel-info">
                                                <h4>Panel {panel.id}</h4>
                                                <p className="panel-p-desc">{panel.description}</p>
                                                {panel.image_path && (
                                                     <button className="action-btn"><RefreshCw size={12} /> Regenerate Panel</button>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="no-data" style={{ padding: '3rem' }}>
                                    <ImageIcon size={48} opacity={0.2} style={{ marginBottom: '1rem' }} />
                                    <p>Waiting for panel list compilation...</p>
                                </div>
                            )}
                        </section>
                    </div>
                )}
            </main>

            <style dangerouslySetInnerHTML={{
                __html: `
        .app-container { display: flex; height: 100vh; overflow: hidden; background: #0d1117; color: #fff; font-family: 'Inter', 'Outfit', sans-serif; }
        .glass { background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; }
        .text-gradient { background: linear-gradient(135deg, #60efff 0%, #00ff87 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        .sidebar { width: 280px; padding: 1.5rem; border-right: 1px solid rgba(255,255,255,0.08); display: flex; flex-direction: column; }
        .sidebar-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2.5rem; }
        .logo-icon { width: 28px; height: 28px; }
        
        .nav-menu { margin-bottom: 1.5rem; display: flex; flex-direction: column; gap: 0.4rem; }
        .nav-item { display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem; border-radius: 8px; cursor: pointer; transition: all 0.3s; opacity: 0.7; }
        .nav-item:hover { background: rgba(255,255,255,0.05); opacity: 1; }
        .nav-item.active { background: rgba(96, 239, 255, 0.1); color: #60efff; opacity: 1; border-left: 3px solid #60efff; }

        .scene-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.3rem; }
        .scene-list h3 { font-size: 0.75rem; text-transform: uppercase; opacity: 0.4; letter-spacing: 0.1em; margin-bottom: 0.8rem; margin-top: 1rem; padding-left: 0.5rem; }
        .scene-item { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; border-radius: 8px; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; border: 1px solid transparent; }
        .scene-item:hover { background: rgba(255,255,255,0.04); }
        .scene-item.active { background: rgba(96, 239, 255, 0.08); color: #60efff; border-color: rgba(96, 239, 255, 0.2); }
        .scene-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 190px; }

        .main-content { flex: 1; padding: 2rem; overflow-y: auto; background: radial-gradient(circle at top right, rgba(96, 239, 255, 0.04), transparent 50%), #090c10; }
        
        .page-header { margin-bottom: 2rem; }
        .page-header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; letter-spacing: -0.02em; }
        .page-header p { opacity: 0.6; font-size: 0.9rem; }

        .projects-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }
        .project-card { padding: 1.5rem; transition: all 0.3s; cursor: pointer; position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between; min-height: 180px; }
        .project-card:hover { transform: translateY(-3px); box-shadow: 0 12px 24px rgba(0,0,0,0.4); border-color: rgba(96, 239, 255, 0.25); }
        .project-card.dashed { border-style: dashed; border-color: rgba(255,255,255,0.15); display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.55; gap: 0.5rem; }
        .project-card.dashed:hover { opacity: 1; border-color: #60efff; border-style: solid; }
        .add-icon { width: 42px; height: 42px; }

        .project-icon { width: 50px; height: 50px; background: rgba(96, 239, 255, 0.08); color: #60efff; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; }
        .status-tag { position: absolute; top: 1.2rem; right: 1.2rem; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; background: rgba(96, 239, 255, 0.08); padding: 0.3rem 0.6rem; border-radius: 20px; color: #60efff; border: 1px solid rgba(96, 239, 255, 0.2); }
        .running-pulse { background: rgba(0, 255, 135, 0.08); color: #00ff87; border-color: rgba(0, 255, 135, 0.2); animation: pulse-animation 2s infinite; }
        .project-meta h3 { margin: 0 0 0.8rem; font-size: 1.05rem; font-weight: 600; }
        .project-stats { display: flex; gap: 1.2rem; font-size: 0.8rem; opacity: 0.7; }
        
        .project-footer { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); display: flex; justify-content: space-between; align-items: center; }
        .timestamp { font-size: 0.7rem; opacity: 0.5; display: flex; align-items: center; gap: 0.4rem; }

        /* CREATE VIEW */
        .create-view { max-width: 800px; margin: 0 auto; padding: 2.5rem; }
        .create-form { display: flex; flex-direction: column; gap: 1.5rem; margin-top: 1.5rem; }
        .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .form-group label { font-size: 0.85rem; font-weight: 600; opacity: 0.8; }
        .form-group input, .form-group select, .form-group textarea { background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; color: #fff; font-family: inherit; font-size: 0.9rem; transition: border 0.3s; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus { border-color: #60efff; outline: none; }
        
        .textarea-header { display: flex; justify-content: space-between; align-items: center; }
        .file-upload-btn { font-size: 0.75rem; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); padding: 0.3rem 0.6rem; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 0.3rem; transition: background 0.3s; }
        .file-upload-btn:hover { background: rgba(255,255,255,0.1); }

        .toggle-container { padding: 1.2rem; display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; }
        .toggle-text h4 { margin: 0 0 0.2rem; font-size: 0.9rem; font-weight: 600; }
        .toggle-text p { margin: 0; font-size: 0.75rem; opacity: 0.5; line-height: 1.4; }
        
        /* Switch slider toggle */
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; flex-shrink: 0; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(255,255,255,0.1); transition: .4s; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; }
        input:checked + .slider { background-color: #00ff87; }
        input:checked + .slider:before { transform: translateX(20px); }
        .slider.round { border-radius: 34px; }
        .slider.round:before { border-radius: 50%; }

        .submit-btn { width: 100%; padding: 1rem; border: none; font-size: 1rem; margin-top: 1rem; border-radius: 8px; transition: all 0.3s; }
        .submit-btn:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(0,255,135,0.2); }
        .spinner { animation: spin 1.5s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* DETAIL VIEW */
        .main-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
        .header-left { display: flex; align-items: center; gap: 1rem; }
        .project-title-area h2 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.2rem; }
        .back-btn { background: none; border: none; color: #fff; cursor: pointer; opacity: 0.6; transition: opacity 0.2s; padding: 0.5rem; display: flex; align-items: center; }
        .back-btn:hover { opacity: 1; }

        .status-badge { display: flex; align-items: center; gap: 0.4rem; font-weight: 600; font-size: 0.75rem; opacity: 0.8; }
        .dot { width: 6px; height: 6px; border-radius: 50%; }
        .dot.green { background: #00ff87; }
        .dot.red { background: #ff4a4a; }
        .pulse { animation: pulse-animation 2s infinite; }
        @keyframes pulse-animation { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

        .global-controls { display: flex; align-items: center; gap: 1rem; }
        .auto-run-status { padding: 0.5rem 0.8rem; display: flex; align-items: center; gap: 0.8rem; font-size: 0.75rem; }
        .auto-run-status strong { color: #60efff; }
        .toggle-btn-small { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.7rem; color: #fff; cursor: pointer; transition: all 0.3s; }
        .toggle-btn-small:hover { background: rgba(255,255,255,0.1); }
        .toggle-btn-small.active { background: rgba(0, 255, 135, 0.1); border-color: rgba(0, 255, 135, 0.3); color: #00ff87; }
        .stop-btn { background: rgba(255, 74, 74, 0.1); border: 1px solid rgba(255, 74, 74, 0.3); color: #ff4a4a; display: flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; cursor: pointer; }

        /* STEPPER COMPONENT */
        .stepper-card { padding: 1.2rem; margin-bottom: 1.5rem; }
        .stepper-card h3 { margin: 0 0 1rem; font-size: 0.85rem; text-transform: uppercase; opacity: 0.4; letter-spacing: 0.08em; }
        .stepper { display: flex; justify-content: space-between; align-items: center; position: relative; }
        .step-node { display: flex; flex-direction: column; align-items: center; text-align: center; position: relative; z-index: 2; flex: 1; }
        .step-circle { width: 32px; height: 32px; border-radius: 50%; background: #161b22; border: 2px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.6rem; transition: all 0.3s; color: rgba(255,255,255,0.4); }
        .step-label h4 { margin: 0; font-size: 0.8rem; font-weight: 600; }
        .step-label p { margin: 0; font-size: 0.65rem; opacity: 0.4; margin-top: 0.1rem; }
        .step-line { position: absolute; height: 2px; width: 100%; left: 50%; top: 16px; background: rgba(255,255,255,0.06); z-index: -1; }
        
        .step-node.completed .step-circle { background: rgba(0, 255, 135, 0.1); border-color: #00ff87; color: #00ff87; }
        .step-node.completed .step-line { background: #00ff87; }
        .step-node.active .step-circle { background: rgba(96, 239, 255, 0.15); border-color: #60efff; color: #60efff; box-shadow: 0 0 12px rgba(96, 239, 255, 0.3); animation: pulse-animation 2s infinite; }
        
        /* ALERT BOX GATE PAUSE */
        .alert-box { padding: 1.2rem; border-color: rgba(96, 239, 255, 0.25); display: flex; align-items: center; gap: 1.2rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(96, 239, 255, 0.03) 100%); }
        .pulse-border { animation: pulse-border-anim 2.5s infinite; }
        @keyframes pulse-border-anim { 0% { border-color: rgba(96,239,255,0.15); } 50% { border-color: rgba(96,239,255,0.45); } 100% { border-color: rgba(96,239,255,0.15); } }
        .alert-icon { color: #60efff; flex-shrink: 0; }
        .alert-text { flex: 1; }
        .alert-text h3 { margin: 0 0 0.3rem; font-size: 0.95rem; font-weight: 600; color: #60efff; }
        .alert-text p { margin: 0; font-size: 0.8rem; opacity: 0.7; line-height: 1.4; }
        .approve-action-btn { background: linear-gradient(135deg, #60efff 0%, #00ff87 100%); border: none; color: #000; padding: 0.6rem 1.2rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; cursor: pointer; transition: transform 0.2s; }
        .approve-action-btn:hover { transform: scale(1.02); }

        .save-status-bar { padding: 0.8rem; text-align: center; font-size: 0.8rem; margin-bottom: 1.2rem; border-color: rgba(0,255,135,0.2); }

        /* GRID STRUCTURE IN DETAIL VIEW */
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }
        .card { padding: 1.5rem; display: flex; flex-direction: column; min-height: 520px; max-height: 700px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.8rem; }
        .card h3 { margin: 0; display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; font-weight: 600; }
        .save-btn { background: rgba(96, 239, 255, 0.08); border: 1px solid rgba(96, 239, 255, 0.2); color: #60efff; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.7rem; cursor: pointer; font-weight: 600; transition: background 0.3s; }
        .save-btn:hover { background: rgba(96, 239, 255, 0.15); }

        /* SCRIPT DETAILS */
        .script-container { flex: 1; overflow-y: auto; padding-right: 0.2rem; }
        .form-group-row { display: flex; gap: 0.8rem; margin-bottom: 0.8rem; }
        .form-group-sub { display: flex; flex-direction: column; gap: 0.4rem; }
        .form-group-sub label { font-size: 0.7rem; font-weight: 600; opacity: 0.5; text-transform: uppercase; letter-spacing: 0.05em; }
        .form-group-sub input, .form-group-sub textarea, .form-group-sub select { background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(255,255,255,0.08); padding: 0.5rem 0.7rem; border-radius: 6px; color: #fff; font-family: inherit; font-size: 0.85rem; }
        
        .flex-1 { flex: 1; }
        .panels-list-header h4 { margin: 1.5rem 0 0.8rem; font-size: 0.8rem; text-transform: uppercase; opacity: 0.5; letter-spacing: 0.05em; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 0.4rem; }
        .panel-editor-card { padding: 1rem; border-color: rgba(255,255,255,0.04); position: relative; margin-bottom: 1rem; }
        .panel-num-tag { font-size: 0.65rem; font-weight: 700; color: #60efff; background: rgba(96,239,255,0.08); display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; margin-bottom: 0.6rem; border: 1px solid rgba(96,239,255,0.15); }
        
        .dialogue-items { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.6rem; }
        .dialogue-items label { font-size: 0.68rem; font-weight: 600; opacity: 0.5; text-transform: uppercase; margin-bottom: 0.2rem; }
        .dialogue-row { display: flex; gap: 0.4rem; align-items: center; }
        .dlg-speaker { width: 90px; flex-shrink: 0; background: rgba(0,0,0,0.3) !important; border-color: rgba(255,255,255,0.05) !important; }
        .dlg-text { flex: 1; background: rgba(0,0,0,0.3) !important; border-color: rgba(255,255,255,0.05) !important; }
        .no-dialogue-label { font-size: 0.75rem; opacity: 0.3; margin: 0; font-style: italic; }

        .approve-stage-btn { margin-top: 1.5rem; width: 100%; padding: 0.8rem; background: rgba(0, 255, 135, 0.1); border: 1px solid rgba(0, 255, 135, 0.3); color: #00ff87; border-radius: 6px; font-size: 0.8rem; font-weight: 700; cursor: pointer; transition: all 0.3s; }
        .approve-stage-btn:hover { background: rgba(0, 255, 135, 0.18); box-shadow: 0 4px 12px rgba(0,255,135,0.1); }

        /* CHARACTERS LIST */
        .char-list-scroll { flex: 1; overflow-y: auto; padding-right: 0.2rem; }
        .char-editor-card { padding: 1.2rem; margin-bottom: 1rem; border-color: rgba(255,255,255,0.04); }
        .char-editor-row { display: flex; gap: 0.6rem; }

        /* VISUAL PLANNING */
        .plan-list-scroll { flex: 1; overflow-y: auto; padding-right: 0.2rem; }
        .plan-helper-txt { font-size: 0.75rem; opacity: 0.5; line-height: 1.4; margin-top: 0; margin-bottom: 1.2rem; }
        .plan-editor-card { padding: 1.2rem; margin-bottom: 1rem; border-color: rgba(255,255,255,0.04); }
        .plan-panel-header { font-size: 0.75rem; font-weight: 700; color: #60efff; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 0.4rem; margin-bottom: 0.8rem; }

        /* NO DATA LABELS */
        .no-data { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem; padding: 2rem; }
        .no-data p { line-height: 1.4; max-width: 210px; margin: 0 0 1.2rem; }
        .loading-bar-placeholder { width: 140px; height: 3px; border-radius: 4px; background: rgba(96,239,255,0.1); position: relative; overflow: hidden; }
        .pulse-bg { animation: pulse-bg-anim 1.5s infinite; }
        @keyframes pulse-bg-anim { 0% { background: rgba(96,239,255,0.08); } 50% { background: rgba(96,239,255,0.25); } 100% { background: rgba(96,239,255,0.08); } }

        /* PANELS PREVIEW */
        .panels-preview-box { width: 100%; display: flex; flex-direction: column; min-height: auto !important; max-height: none !important; }
        .panels-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1.5rem; margin-top: 0.5rem; }
        .panel-card { padding: 1rem; border-radius: 12px; display: flex; flex-direction: column; gap: 0.8rem; justify-content: space-between; }
        .panel-placeholder { height: 210px; background: #0c0e12; border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid rgba(255,255,255,0.04); }
        .panel-placeholder img { width: 100%; height: 100%; object-fit: cover; }
        .render-waiting { text-align: center; color: rgba(255,255,255,0.2); font-size: 0.75rem; display: flex; flex-direction: column; align-items: center; gap: 0.5rem; }
        .panel-info h4 { margin: 0 0 0.3rem; font-size: 0.9rem; font-weight: 600; }
        .panel-p-desc { font-size: 0.75rem; opacity: 0.6; margin: 0; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .action-btn { background: rgba(96, 239, 255, 0.08); border: 1px solid rgba(96, 239, 255, 0.2); color: #60efff; display: flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.6rem; border-radius: 5px; font-size: 0.7rem; cursor: pointer; transition: background 0.3s; margin-top: 0.4rem; }
        .action-btn:hover { background: rgba(96, 239, 255, 0.15); }

        /* Animation */
        .animate-fade-in { animation: fadeIn 0.4s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
      `}} />
        </div>
    );
}

export default App;
