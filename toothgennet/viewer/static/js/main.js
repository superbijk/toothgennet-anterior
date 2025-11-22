import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// Scene Setup
const container = document.getElementById('container');
// Debug: Log devicePixelRatio and wheel events
console.log(`[Debug] window.devicePixelRatio: ${window.devicePixelRatio}`);
container.addEventListener('wheel', (e) => {
    console.log(`[Debug] Wheel DeltaY: ${e.deltaY}`);
});
const staticHeatmapContainer = document.getElementById('staticHeatmapContainer');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setScissorTest(true);
container.appendChild(renderer.domElement);

// State
let scenes = []; // Array of { scene, camera, controls, element }
let animationId = null;
let currentGrid = { rows: 1, cols: 1 }; // Track grid layout for resize

// --- Layout & Resize Logic ---

function resizeRenderer() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    
    if (scenes.length > 0) {
        const cellWidth = width / currentGrid.cols;
        const cellHeight = height / currentGrid.rows;
        
        scenes.forEach(s => {
            const { row, col } = s.userData;
            
            s.rect.x = col * cellWidth;
            s.rect.y = row * cellHeight;
            s.rect.width = cellWidth;
            s.rect.height = cellHeight;
            
            s.camera.aspect = cellWidth / cellHeight;
            s.camera.updateProjectionMatrix();

            // Update label position
            if (s.label) {
                s.label.style.left = (s.rect.x + 5) + 'px';
                s.label.style.top = (s.rect.y + 5) + 'px';
            }
        });

        // Synchronize heatmap cells
        const heatmapCells = document.querySelectorAll('.heatmap-cell');
        heatmapCells.forEach(cell => {
            cell.style.width = `${cellWidth}px`;
            cell.style.minWidth = `${cellWidth}px`;
        });
    }
}

function setLayout(mode) {
    if (mode === 'inspect') {
        staticHeatmapContainer.style.height = '0';
    } else if (mode === 'interpolate') {
        staticHeatmapContainer.style.height = '30%';
    }
    
    // Animate resize during transition (approx 350ms to match CSS)
    let start = Date.now();
    let interval = setInterval(() => {
        resizeRenderer();
        if (Date.now() - start > 350) clearInterval(interval);
    }, 16);
}

// Window resize
window.addEventListener('resize', resizeRenderer);


// --- Scene Management ---

function clearScenes() {
    scenes.forEach(s => {
        if (s.controls) {
            s.controls.dispose();
        }
        if (s.label && s.label.parentNode) {
            s.label.parentNode.removeChild(s.label);
        }
    });
    scenes = [];
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    currentGrid = { rows: 1, cols: 1 };
    
    // Clear static heatmaps
    staticHeatmapContainer.innerHTML = '';
}

function createScene(x, y, width, height, enableControls = false) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Ensure width and height are valid to prevent division by zero or NaN aspect ratio
    const validWidth = width > 0 ? width : 1;
    const validHeight = height > 0 ? height : 1;
    const aspect = validWidth / validHeight;

    console.log(`[createScene] x=${x}, y=${y}, width=${width}, height=${height}, validWidth=${validWidth}, validHeight=${validHeight}, aspect=${aspect}, enableControls=${enableControls}`);

    const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.up.set(0, 0, 1); // Z-up coordinate system
    camera.position.set(0, -2.5, 0.5); // View from -Y (Facial)
    
    console.log(`[createScene] Camera initial position: (${camera.position.x}, ${camera.position.y}, ${camera.position.z})`);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x808080);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, -1, 1);
    scene.add(directionalLight);
    const dl2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dl2.position.set(0, 1, 0.5);
    scene.add(dl2);

    // Controls - only create if this is the master scene
    let controls = null;
    if (enableControls) {
        // Fix devicePixelRatio if < 1 (OrbitControls uses bitwise OR which truncates decimals to 0)
        // e.g., 0.9375 | 0 = 0, causing division by zero in zoom calculation
        if (!window.devicePixelRatio || window.devicePixelRatio < 1) {
            const oldValue = window.devicePixelRatio;
            window.devicePixelRatio = 1;
            console.log(`[Fix] devicePixelRatio was ${oldValue}, set to 1 (OrbitControls bitwise truncation fix)`);
        }

        controls = new OrbitControls(camera, renderer.domElement);
        // Set minimum distance to prevent camera from collapsing into target
        controls.minDistance = 0.1;
        // Set maximum distance to prevent zooming too far out
        controls.maxDistance = 10.0;
        // Set target to origin to prevent camera from collapsing into object
        controls.target.set(0, 0, 0);
        controls.update();
        console.log(`[createScene] Controls created. Target: (${controls.target.x}, ${controls.target.y}, ${controls.target.z})`);
    }
    
    // Create label element
    const label = document.createElement('div');
    label.className = 'scene-label';
    label.style.display = 'none'; // Hidden by default
    label.style.position = 'absolute';
    label.style.color = 'black';
    label.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    label.style.padding = '2px 6px';
    label.style.fontSize = '12px';
    label.style.fontFamily = 'sans-serif';
    label.style.pointerEvents = 'none';
    label.style.borderRadius = '4px';
    label.style.border = '1px solid #ccc';
    label.style.zIndex = '10';
    container.appendChild(label);

    // Initial position update
    label.style.left = (x + 5) + 'px';
    label.style.top = (y + 5) + 'px';

    return { 
        scene, 
        camera, 
        controls, 
        rect: { x, y, width, height },
        label: label,
        userData: {}
    };
}

function render() {
    animationId = requestAnimationFrame(render);

    // Sync cameras if we have multiple scenes (grid view)
    if (scenes.length > 1) {
        const master = scenes[0];
        if (master.controls) {
            master.controls.update();
            
            for (let i = 1; i < scenes.length; i++) {
                const s = scenes[i];
                s.camera.position.copy(master.camera.position);
                s.camera.rotation.copy(master.camera.rotation);
                s.camera.zoom = master.camera.zoom;
                if (master.controls) {
                    s.camera.updateProjectionMatrix();
                }
            }
        }
    } else if (scenes.length === 1 && scenes[0].controls) {
        scenes[0].controls.update();
    }

    const canvasHeight = renderer.domElement.height;
    
    scenes.forEach(s => {
        const { x, y, width, height } = s.rect;
        // Scissor y is from bottom
        renderer.setViewport(x, canvasHeight - y - height, width, height);
        renderer.setScissor(x, canvasHeight - y - height, width, height);
        renderer.render(s.scene, s.camera);
    });
}

// Start loop
render();

// Debug: Log camera state every 1s
setInterval(() => {
    if (scenes.length > 0) {
        const s = scenes[0];
        if (s.controls) {
            const p = s.camera.position;
            const t = s.controls.target;
            const dist = p.distanceTo(t);
            console.log(`[Debug] Camera Pos: (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) | Target: (${t.x.toFixed(3)}, ${t.y.toFixed(3)}, ${t.z.toFixed(3)}) | Distance: ${dist.toFixed(3)} | isNaN: ${isNaN(dist)}`);
        }
    }
}, 1000);


// --- Visualization Helpers ---

function populateScenes(data) {
    clearScenes();
    
    const numSteps = data.mesh_urls.length;
    const cols = numSteps;
    const rows = 2; // Meshes, Points
    currentGrid = { rows, cols };
    
    const threeD_width = container.clientWidth;
    const threeD_height = container.clientHeight;
    const cellWidth = threeD_width / cols;
    const cellHeight = threeD_height / rows; 
    
    const loader = new OBJLoader();
    
    // Row 1: Meshes
    for (let i = 0; i < numSteps; i++) {
        const s = createScene(i * cellWidth, 0, cellWidth, cellHeight, i === 0);
        s.userData = { row: 0, col: i };
        
        loader.load(data.mesh_urls[i], (obj) => {
            obj.traverse(child => {
                if (child.isMesh) {
                    child.material = new THREE.MeshStandardMaterial({ 
                        color: document.getElementById('meshColor').value,
                        flatShading: false,
                        roughness: parseFloat(document.getElementById('meshRoughness').value),
                        metalness: 0.0
                    });
                }
            });
            s.scene.add(obj);
        });
        scenes.push(s);
    }
    
    // Row 2: Points
    for (let i = 0; i < numSteps; i++) {
        const s = createScene(i * cellWidth, cellHeight, cellWidth, cellHeight, false);
        s.userData = { row: 1, col: i };
        
        const ptsData = data.point_clouds[i];
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(ptsData.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        const material = new THREE.PointsMaterial({ 
            size: parseFloat(document.getElementById('pointSize').value),
            color: document.getElementById('pointColor').value,
            vertexColors: false 
        });
        const points = new THREE.Points(geometry, material);
        s.scene.add(points);
        scenes.push(s);
    }

    // Static Heatmaps
    if (data.heatmap_urls && data.heatmap_urls.length > 0) {
        data.heatmap_urls.forEach(url => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.style.width = `${cellWidth}px`;
            cell.style.minWidth = `${cellWidth}px`;
            
            const img = document.createElement('img');
            img.src = url;
            cell.appendChild(img);
            staticHeatmapContainer.appendChild(cell);
        });
    }
}


// --- API Calls ---

async function loadMesh(index) {
    try {
        setLayout('inspect');
        const response = await fetch(`/api/mesh/${index}`);
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }
        
        if (data.total_samples !== undefined) {
            document.getElementById('maxIndexLabel').innerText = `(0 - ${data.total_samples - 1})`;
        }

        clearScenes();
        currentGrid = { rows: 1, cols: 1 };
        const s = createScene(0, 0, container.clientWidth, container.clientHeight, true);
        s.userData = { row: 0, col: 0 };
        
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(data.points.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        const material = new THREE.PointsMaterial({ 
            color: document.getElementById('pointColor').value,
            size: parseFloat(document.getElementById('pointSize').value),
            vertexColors: false 
        });
        const points = new THREE.Points(geometry, material);
        s.scene.add(points);
        
        geometry.computeBoundingSphere();
        console.log(`[loadMesh] Bounding sphere - center: (${geometry.boundingSphere?.center.x}, ${geometry.boundingSphere?.center.y}, ${geometry.boundingSphere?.center.z}), radius: ${geometry.boundingSphere?.radius}`);
        
        // DO NOT set target to bounding sphere center - this causes camera to collapse
        // The camera is already positioned correctly relative to origin (0,0,0)
        // Keep controls.target at (0,0,0)
        if (s.controls) {
            console.log(`[loadMesh] Controls target: (${s.controls.target.x}, ${s.controls.target.y}, ${s.controls.target.z})`);
            console.log(`[loadMesh] Camera position: (${s.camera.position.x}, ${s.camera.position.y}, ${s.camera.position.z})`);
        }
        
        scenes.push(s);
        document.getElementById('meshInfo').innerText = `Category: ${data.category}`;
    } catch (err) {
        console.error(err);
        alert("Failed to load mesh");
    }
}

async function runInterpolation() {
    const idx1 = document.getElementById('interpStart').value;
    const idx2 = document.getElementById('interpEnd').value;
    const steps = document.getElementById('interpSteps').value;
    const regenerate = document.getElementById('interpRegenerate').checked;

    try {
        const response = await fetch('/api/interpolate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx1, idx2, steps: parseInt(steps), regenerate })
        });
        const data = await response.json();
        if (data.error) { alert(data.error); return; }
        if (data.success) {
            setLayout('interpolate');
            populateScenes(data);
        }
    } catch (err) {
        console.error(err);
        alert("Interpolation failed");
    }
}

async function runGenerate() {
    const samples = document.getElementById('genSamples').value;
    // Removed genRegenerate as requested
    const saveObj = document.getElementById('genSaveObj').checked;

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                num_samples: parseInt(samples), 
                regenerate: true, // Always regenerate for random samples
                save_obj: saveObj
            })
        });
        const data = await response.json();
        if (data.error) { alert(data.error); return; }
        if (data.success) {
            setLayout('interpolate');
            populateScenes(data);
        }
    } catch (err) {
        console.error(err);
        alert("Generation failed");
    }
}

async function clearGeneration() {
    if (!confirm("Are you sure you want to delete all generated files?")) return;
    try {
        const response = await fetch('/api/clear_generation', { method: 'POST' });
        const data = await response.json();
        if (data.error) alert(data.error);
        else alert(data.message);
    } catch(err) {
        console.error(err);
        alert("Failed to clear generation cache");
    }
}

async function generateCut() {
    const cutType = document.getElementById('cutType').value;
    try {
        const response = await fetch('/api/generate_cut', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cut_type: cutType })
        });
        const data = await response.json();
        if (data.error) return;
        document.getElementById('cutCondition').value = data.cut_formula;
    } catch (err) {}
}

async function previewCut() {
    const sampleIdx = document.getElementById('restoreIdx').value;
    const cutCondition = document.getElementById('cutCondition').value;
    
    try {
        const response = await fetch('/api/preview_cut', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample_idx: sampleIdx, cut_condition: cutCondition })
        });
        const data = await response.json();
        if (data.error) { alert(data.error); return; }
        
        setLayout('inspect');
        clearScenes();
        currentGrid = { rows: 1, cols: 1 };
        const s = createScene(0, 0, container.clientWidth, container.clientHeight, true);
        s.userData = { row: 0, col: 0 };
        
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(data.points.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        const count = data.points.length;
        const cutCount = data.cut_count;
        const colors = new Float32Array(count * 3);
        
        const colorOriginal = new THREE.Color(document.getElementById('pointColor').value);
        const colorCut = new THREE.Color(0xff0000); // Red for cut part
        
        for (let i = 0; i < count; i++) {
            // First cutCount points are kept (Red), rest are excluded (Original color)
            // Based on backend logic: kept_points comes first.
            // Wait, user asked to "put red in visible part".
            // Visible = kept.
            // So first cutCount points (kept) -> Red.
            // Rest (excluded) -> Original.
            const color = i < cutCount ? colorCut : colorOriginal;
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({ 
            size: parseFloat(document.getElementById('pointSize').value),
            vertexColors: true 
        });
        const points = new THREE.Points(geometry, material);
        s.scene.add(points);
        
        geometry.computeBoundingSphere();
        console.log(`[previewCut] Bounding sphere - center: (${geometry.boundingSphere?.center.x}, ${geometry.boundingSphere?.center.y}, ${geometry.boundingSphere?.center.z}), radius: ${geometry.boundingSphere?.radius}`);
        
        // DO NOT set target to bounding sphere center
        if (s.controls) {
            console.log(`[previewCut] Controls target: (${s.controls.target.x}, ${s.controls.target.y}, ${s.controls.target.z})`);
            console.log(`[previewCut] Camera position: (${s.camera.position.x}, ${s.camera.position.y}, ${s.camera.position.z})`);
        }
        scenes.push(s);
        
    } catch (err) {
        console.error(err);
        alert("Preview failed");
    }
}

function populateRestorationScenes(data) {
    clearScenes();
    
    const inputPoints = data.input_points;
    const missingPoints = data.missing_points;
    const restoredSteps = data.restored_steps;
    
    const numSteps = restoredSteps.length;
    const totalPanels = 2 + numSteps; // Ref, Input, Steps...
    
    // Grid layout: 1 row
    const cols = totalPanels;
    const rows = 1;
    currentGrid = { rows, cols };
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    const cellWidth = width / cols;
    const cellHeight = height; // Full height for 1 row
    
    const pointSize = parseFloat(document.getElementById('pointSize').value);
    const colorOriginal = new THREE.Color(document.getElementById('pointColor').value);
    const colorCut = new THREE.Color(0xff0000); // Red
    const colorRestored = new THREE.Color(0xffa500); // Orange
    
    // Helper to create point cloud from points and colors
    const createPC = (points, colorObj) => {
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(points.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        const count = points.length;
        const colors = new Float32Array(count * 3);
        for(let i=0; i<count; i++) {
            colors[i*3] = colorObj.r;
            colors[i*3+1] = colorObj.g;
            colors[i*3+2] = colorObj.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        return geometry;
    };

    // Panel 1: Reference (Input + Missing)
    {
        const s = createScene(0, 0, cellWidth, cellHeight, true);
        s.userData = { row: 0, col: 0 };
        s.label.innerText = "Ref. Point Cloud";
        s.label.style.display = 'block';
        
        // Combine points
        const allPoints = [...inputPoints, ...missingPoints];
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(allPoints.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        if (vertices.length > 0) {
            geometry.computeBoundingSphere();
            console.log(`[populateRestorationScenes Panel1] Bounding sphere - center: (${geometry.boundingSphere?.center.x}, ${geometry.boundingSphere?.center.y}, ${geometry.boundingSphere?.center.z}), radius: ${geometry.boundingSphere?.radius}`);
            // DO NOT set target to bounding sphere center - keep at origin
        }
        
        const count = allPoints.length;
        const inputCount = inputPoints.length;
        const colors = new Float32Array(count * 3);
        
        for(let i=0; i<count; i++) {
            const color = i < inputCount ? colorOriginal : colorCut;
            colors[i*3] = color.r;
            colors[i*3+1] = color.g;
            colors[i*3+2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true });
        s.scene.add(new THREE.Points(geometry, material));
        scenes.push(s);
    }
    
    // Panel 2: Input Only
    {
        const s = createScene(cellWidth, 0, cellWidth, cellHeight, false);
        s.userData = { row: 0, col: 1 };
        s.label.innerText = "Input Point Cloud";
        s.label.style.display = 'block';
        
        const geometry = createPC(inputPoints, colorOriginal);
        const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true });
        s.scene.add(new THREE.Points(geometry, material));
        scenes.push(s);
    }
    
    // Panel 3+: Restored Steps
    console.log(`Populating ${numSteps} restoration steps`);
    for(let i=0; i<numSteps; i++) {
        const s = createScene((i + 2) * cellWidth, 0, cellWidth, cellHeight, false);
        s.userData = { row: 0, col: i + 2 };
        s.label.innerText = `Latent Optimization Step ${i+1}/${numSteps}`;
        s.label.style.display = 'block';
        
        const stepPoints = restoredSteps[i];
        console.log(`Step ${i}: ${stepPoints.length} points`);
        
        // If stepPoints is empty (e.g. early optimization steps might not generate points in the cut region yet),
        // we still want to show the input points.
        const allPoints = [...inputPoints, ...stepPoints];
        
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(allPoints.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        // Only compute bounding sphere if we have points
        if (vertices.length > 0) {
            geometry.computeBoundingSphere();
        }
        
        const count = allPoints.length;
        const inputCount = inputPoints.length;
        const colors = new Float32Array(count * 3);
        
        for(let j=0; j<count; j++) {
            const color = j < inputCount ? colorOriginal : colorRestored;
            colors[j*3] = color.r;
            colors[j*3+1] = color.g;
            colors[j*3+2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true });
        s.scene.add(new THREE.Points(geometry, material));
        scenes.push(s);
    }
    
    // Static Images
    if (data.image_urls && data.image_urls.length > 0) {
        staticHeatmapContainer.innerHTML = '';
        staticHeatmapContainer.style.display = 'flex';
        staticHeatmapContainer.style.justifyContent = 'center';
        staticHeatmapContainer.style.alignItems = 'center';
        staticHeatmapContainer.style.gap = '20px';
        
        data.image_urls.forEach(url => {
            const img = document.createElement('img');
            img.src = url;
            img.style.height = '95%';
            img.style.width = 'auto';
            img.style.maxWidth = '45%';
            img.style.objectFit = 'contain';
            staticHeatmapContainer.appendChild(img);
        });
    }
}

async function runRestoration() {
    const cutType = document.getElementById('cutType').value;
    const sampleIdx = document.getElementById('restoreIdx').value;
    const cutCondition = document.getElementById('cutCondition').value;
    const steps = document.getElementById('restoreSteps').value;
    const lr = document.getElementById('restoreLR').value;
    const stepSize = document.getElementById('restoreStep').value;
    const gamma = document.getElementById('restoreGamma').value;
    const vizSteps = document.getElementById('restoreVizSteps').value;

    try {
        const response = await fetch('/api/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                cut_type: cutType, 
                sample_idx: sampleIdx, 
                cut_condition: cutCondition,
                num_iterations: parseInt(steps),
                learning_rate: parseFloat(lr),
                step_size: parseInt(stepSize),
                gamma: parseFloat(gamma),
                viz_steps: parseInt(vizSteps)
            })
        });
        const data = await response.json();
        if (data.error) { alert(data.error); return; }

        setLayout('interpolate');
        populateRestorationScenes(data);
        
        // alert(data.message); // Removed alert as requested

    } catch (err) {
        console.error(err);
        alert("Restoration failed");
    }
}


// --- UI Event Handlers ---

function updateVisuals() {
    const meshColor = document.getElementById('meshColor').value;
    const meshRoughness = parseFloat(document.getElementById('meshRoughness').value);
    const pointColorHex = document.getElementById('pointColor').value;
    const pointSize = parseFloat(document.getElementById('pointSize').value);
    let numPointsInput = document.getElementById('numPoints').value;
    let numPointsToDraw;

    scenes.forEach(s => {
        s.scene.traverse(child => {
            if (child.isMesh) {
                child.material.color.set(meshColor);
                child.material.roughness = meshRoughness;
            }
            if (child.isPoints) {
                child.material.size = pointSize;
                if (!child.material.vertexColors) {
                    child.material.color.set(pointColorHex);
                }
                child.material.needsUpdate = true;
                
                if (!numPointsInput || isNaN(parseInt(numPointsInput))) {
                    numPointsToDraw = child.geometry.attributes.position.count;
                } else {
                    numPointsToDraw = parseInt(numPointsInput);
                }
                if (child.geometry) {
                    numPointsToDraw = Math.min(numPointsToDraw, child.geometry.attributes.position.count);
                    child.geometry.setDrawRange(0, numPointsToDraw);
                }
            }
        });
    });
}

function saveSettings() {
    const settings = {
        meshColor: document.getElementById('meshColor').value,
        meshRoughness: document.getElementById('meshRoughness').value,
        pointColor: document.getElementById('pointColor').value,
        numPoints: document.getElementById('numPoints').value, 
        pointSize: document.getElementById('pointSize').value 
    };
    document.cookie = `visualSettings=${encodeURIComponent(JSON.stringify(settings))}; path=/; max-age=31536000`;
}

function loadSettings() {
    const cookies = document.cookie.split(';');
    const settingsCookie = cookies.find(c => c.trim().startsWith('visualSettings='));
    if (settingsCookie) {
        try {
            const jsonStr = decodeURIComponent(settingsCookie.split('=')[1]);
            const settings = JSON.parse(jsonStr);
            
            if (settings.meshColor !== undefined) document.getElementById('meshColor').value = settings.meshColor;
            if (settings.meshRoughness !== undefined) document.getElementById('meshRoughness').value = settings.meshRoughness;
            if (settings.pointColor !== undefined) document.getElementById('pointColor').value = settings.pointColor;
            if (settings.numPoints !== undefined) document.getElementById('numPoints').value = settings.numPoints;
            if (settings.pointSize !== undefined) document.getElementById('pointSize').value = settings.pointSize;
            
            updateVisuals();
        } catch (e) {}
    }
}

document.getElementById('meshColor').addEventListener('input', updateVisuals);
document.getElementById('meshRoughness').addEventListener('input', updateVisuals);
document.getElementById('pointColor').addEventListener('input', updateVisuals);

document.getElementById('applyNumPoints').addEventListener('click', updateVisuals);
document.getElementById('applyPointSize').addEventListener('click', updateVisuals);

document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
document.getElementById('loadSettingsBtn').addEventListener('click', loadSettings);

document.getElementById('loadMeshBtn').addEventListener('click', () => {
    // if (animationId) cancelAnimationFrame(animationId); // Don't stop the loop
    loadMesh(document.getElementById('meshIndex').value);
});

document.getElementById('interpolateBtn').addEventListener('click', runInterpolation);
document.getElementById('generateBtn').addEventListener('click', runGenerate);
document.getElementById('clearGenBtn').addEventListener('click', clearGeneration);
document.getElementById('restoreBtn').addEventListener('click', runRestoration);
document.getElementById('previewCutBtn').addEventListener('click', previewCut);
document.getElementById('generateCutBtn').addEventListener('click', generateCut);

// Initial Load
generateCut();
loadMesh(0);