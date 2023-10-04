// static/main.js

const sliders = document.querySelectorAll('input[type="range"]');
sliders.forEach((slider) => {
    slider.addEventListener('input', () => {
        const value = slider.value;
        const label = document.querySelector(`label[for="${slider.id}"]`);
        label.textContent = `${slider.id}: ${value}`;
    });
});