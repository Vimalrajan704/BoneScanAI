
// DOM Elements
const uploadArea = document.getElementById('upload-area');
const uploadContent = document.getElementById('upload-content');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const replaceText = document.getElementById('replace-text');
const selectImageBtn = document.getElementById('select-image-btn');
const analyzeContainer = document.getElementById('analyze-container');
const analyzeBtn = document.getElementById('analyze-btn');
const resultCard = document.getElementById('result-card');
const resultContent = document.getElementById('result-content');
const loadingIndicator = document.getElementById('loading-indicator');
const progressBar = document.getElementById('progress-bar');
const progressPercentage = document.getElementById('progress-percentage');
const toast = document.getElementById('toast');
const toastTitle = document.getElementById('toast-title');
const toastMessage = document.getElementById('toast-message');

// Global variables
let uploadedFile = null;

// Functions
function scrollToUpload() {
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
}

function showToast(title, message, type = 'info') {
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    
    // Set toast color based on type
    toast.className = 'toast show';
    
    // Add type-specific styling if needed
    if (type === 'error') {
        toast.style.borderLeft = '4px solid var(--error)';
    } else if (type === 'success') {
        toast.style.borderLeft = '4px solid var(--success)';
    } else {
        toast.style.borderLeft = '4px solid var(--primary)';
    }
    
    // Hide toast after 5 seconds
    setTimeout(() => {
        toast.className = 'toast';
    }, 5000);
}

function handleFileUpload(file) {
    // Check if file is an image
    if (!file.type.match('image.*')) {
        showToast('Invalid file type', 'Please upload an image file.', 'error');
        return;
    }
    
    // Store the file
    uploadedFile = file;
    
    // Generate image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        // Show preview
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
        replaceText.style.display = 'block';
        
        // Show analyze button
        analyzeContainer.style.display = 'block';
        
        showToast('X-ray uploaded', 'Your image has been successfully uploaded for analysis.');
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!uploadedFile) return;

    resultContent.innerHTML = '';
    loadingIndicator.style.display = 'block';
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const titleElement = resultCard.querySelector('h3');
    titleElement.innerHTML = 'Analysis Results <span class="result-badge loading-badge">Analyzing...</span>';

    const formData = new FormData();
    formData.append('image', uploadedFile);

    try {
        const response = await fetch('https://bonescanai-api.onrender.com/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('Failed to analyze image');

        const result = await response.json(); // result = { detection, message, image_base64 }
        const imageUrl = `data:image/jpeg;base64,${result.image_base64}`;

        loadingIndicator.style.display = 'none';
        analyzeBtn.textContent = 'Analyze X-ray';
        analyzeBtn.disabled = false;

        titleElement.innerHTML = 'Analysis Results <span class="result-badge success-badge">Fracture Detection Complete</span>';

        if (!result.detection) {
            resultContent.innerHTML = `
                <div style="text-align: center;">
                    <h4>No Fracture Detected</h4>
                    <img src="${imageUrl}" alt="No Detection" style="max-height: 300px; border-radius: 10px; border: 1px solid #ccc;" />
                </div>
                <p class="result-footnote">${result.message} This is an AI analysis – please confirm with a medical professional.</p>
            `;
        } else {
            resultContent.innerHTML = `
                <div style="text-align: center;">
                    <h4>Annotated Image</h4>
                    <img id="annotated-image" src="${imageUrl}" alt="YOLOv8 Annotated" style="max-height: 300px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;" />
                </div>
                <p class="result-footnote">${result.message}This is an AI analysis - Please confirm with a medical professional.</p>
            `;

            const annotatedImage = document.getElementById('annotated-image');
            annotatedImage.addEventListener('click', () => toggleImageZoom(annotatedImage.src));
        }
    } catch (error) {
        console.error('Error analyzing image:', error);
        loadingIndicator.style.display = 'none';
        analyzeBtn.textContent = 'Analyze X-ray';
        analyzeBtn.disabled = false;

        resultContent.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
        showToast('Error', error.message, 'error');
    }
}



function toggleImageZoom(src) {
    const modal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');

    const isVisible = !modal.classList.contains('hidden');

    if (isVisible) {
        modal.classList.add('hidden');
        document.body.classList.remove('blurred');
    } else {
        modalImage.src = src;
        modal.classList.remove('hidden');
        document.body.classList.add('blurred');
    }
}





// For testing when API is not available
function getDemoResult() {
    const randomValue = Math.random();
    return {
        hasFracture: randomValue > 0.5,
        confidence: 0.7 + (randomValue * 0.25),
        region: randomValue > 0.5 ? "Distal radius" : "Tibia",
        description: "Analysis indicates potential fracture line visible along the bone cortex. Recommend clinical correlation."
    };
}

// Event listeners
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragging');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragging');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragging');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

analyzeBtn.addEventListener('click', analyzeImage);

// Initialize results card
const titleElement = resultCard.querySelector('h3');
if (!titleElement) {
    const titleH3 = document.createElement('h3');
    titleH3.textContent = 'Analysis Results';
    resultCard.insertBefore(titleH3, resultContent);
}
document.getElementById('image-modal').addEventListener('click', () => {
    document.getElementById('image-modal').classList.add('hidden');
    document.body.classList.remove('blurred');
});

