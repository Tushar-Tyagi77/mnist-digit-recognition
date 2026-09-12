const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");


// Drawing settings
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";


let isDrawing = false;


// Get mouse/touch position
function getPosition(event) {

    const rect = canvas.getBoundingClientRect();

    let x;
    let y;

    if (event.touches) {
        x = event.touches[0].clientX - rect.left;
        y = event.touches[0].clientY - rect.top;
    } else {
        x = event.clientX - rect.left;
        y = event.clientY - rect.top;
    }

    // Scale position according to canvas size
    x = x * (canvas.width / rect.width);
    y = y * (canvas.height / rect.height);

    return { x, y };
}


// Start drawing
function startDrawing(event) {

    event.preventDefault();

    isDrawing = true;

    const position = getPosition(event);

    ctx.beginPath();

    ctx.moveTo(
        position.x,
        position.y
    );
}


// Draw
function draw(event) {

    event.preventDefault();

    if (!isDrawing) {
        return;
    }

    const position = getPosition(event);

    ctx.lineTo(
        position.x,
        position.y
    );

    ctx.stroke();
}


// Stop drawing
function stopDrawing(event) {

    if (event) {
        event.preventDefault();
    }

    isDrawing = false;

    ctx.closePath();
}


// Mouse events
canvas.addEventListener(
    "mousedown",
    startDrawing
);

canvas.addEventListener(
    "mousemove",
    draw
);

canvas.addEventListener(
    "mouseup",
    stopDrawing
);

canvas.addEventListener(
    "mouseleave",
    stopDrawing
);


// Touch events
canvas.addEventListener(
    "touchstart",
    startDrawing
);

canvas.addEventListener(
    "touchmove",
    draw
);

canvas.addEventListener(
    "touchend",
    stopDrawing
);


// Clear canvas
clearButton.addEventListener(
    "click",
    function () {

        ctx.fillStyle = "black";

        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        predictionText.textContent = "-";
        confidenceText.textContent = "-";
    }
);


// Predict digit
predictButton.addEventListener(
    "click",
    async function () {

        // Convert canvas to image
        canvas.toBlob(
            async function (blob) {

                const formData = new FormData();

                formData.append(
                    "file",
                    blob,
                    "digit.png"
                );


                try {

                    const response = await fetch(
                        "https://mnist-digit-recognition-7p0z.onrender.com/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                    const result = await response.json();


                    if (!response.ok) {

                        throw new Error(
                            "Prediction failed"
                        );

                    }


                    predictionText.textContent =
                        result.predicted_digit;

                    confidenceText.textContent =
                        result.confidence + "%";


                } catch (error) {

                    console.error(error);

                    predictionText.textContent =
                        "Error";

                    confidenceText.textContent =
                        "-";

                    alert(
                        "Could not connect to the FastAPI server."
                    );
                }

            },
            "image/png"
        );
    }
);