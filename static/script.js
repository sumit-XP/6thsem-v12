// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const detectBtn = document.getElementById('detectBtn');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const totalDetections = document.getElementById('totalDetections');
const statsGrid = document.getElementById('statsGrid');
const detectionsList = document.getElementById('detectionsList');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFile(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, WEBP, BMP)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    // Hide results from previous detection
    resultsSection.style.display = 'none';
}

// Detect button click handler
detectBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading state
    const btnText = detectBtn.querySelector('.btn-text');
    const spinner = detectBtn.querySelector('.spinner');
    btnText.textContent = 'Detecting...';
    spinner.style.display = 'block';
    detectBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Detection failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        // Reset button state
        btnText.textContent = 'Detect Behavior';
        spinner.style.display = 'none';
        detectBtn.disabled = false;
    }
});

// Display detection results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Display annotated image
    resultImage.src = data.result_image + '?t=' + new Date().getTime();

    // Display total detections
    totalDetections.textContent = data.total_detections;

    // Display behavior counts
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${data.total_detections}</div>
            <div class="stat-label">Total Detections</div>
        </div>
    `;

    // Add behavior count cards
    const behaviorColors = {
        'Drinking': '#3b82f6',
        'Eating': '#10b981',
        'Sitting': '#f59e0b',
        'Standing': '#8b5cf6'
    };

    for (const [behavior, count] of Object.entries(data.behavior_counts)) {
        const color = behaviorColors[behavior] || '#6366f1';
        statsGrid.innerHTML += `
            <div class="stat-card">
                <div class="stat-value" style="background: linear-gradient(135deg, ${color}, ${color}90); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">${count}</div>
                <div class="stat-label">${behavior}</div>
            </div>
        `;
    }

    // Display detections list
    detectionsList.innerHTML = '';

    if (data.detections.length === 0) {
        detectionsList.innerHTML = '<p style="text-align: center; opacity: 0.7;">No behaviors detected in this image.</p>';
    } else {
        data.detections.forEach((detection, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            detectionItem.innerHTML = `
                <div>
                    <div class="detection-class">${index + 1}. ${detection.class}</div>
                    <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.25rem;">
                        Box: [${detection.bbox.map(x => x.toFixed(0)).join(', ')}]
                    </div>
                </div>
                <div class="detection-confidence">${detection.confidence}%</div>
            `;
            detectionsList.appendChild(detectionItem);
        });
    }
}

// Add some initial animation
document.addEventListener('DOMContentLoaded', () => {
    console.log('🐄 Cow Behavior Detection Dashboard Loaded');
});
