const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');

imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = (event) => {
        imagePreview.innerHTML = `<img src="${event.target.result}" alt="Selected Image" />`;
    };

    reader.readAsDataURL(file);
});