document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Elements
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const resultPrice = document.getElementById('resultPrice');
    const predictBtn = document.getElementById('predictBtn');

    // Show loading state
    btnText.style.display = 'none';
    loader.style.display = 'block';
    predictBtn.disabled = true;
    resultPrice.innerText = 'Calculating...';

    // Gather data
    const data = {
        area: document.getElementById('area').value,
        bedrooms: document.getElementById('bedrooms').value,
        bathrooms: document.getElementById('bathrooms').value,
        location_score: document.getElementById('location_score').value
    };

    try {
        // We simulate a slight delay for the loading animation to look professional
        await new Promise(resolve => setTimeout(resolve, 800));

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.error) {
            resultPrice.innerText = 'Error';
            alert('Prediction failed: ' + result.error);
        } else {
            resultPrice.innerText = result.price;
        }

    } catch (error) {
        resultPrice.innerText = 'Error';
        alert('Network error. Please try again.');
        console.error(error);
    } finally {
        // Hide loading state
        btnText.style.display = 'block';
        loader.style.display = 'none';
        predictBtn.disabled = false;
    }
});
