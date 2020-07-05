let canvas = document.querySelector('#canvas');
canvas.setAttribute('width', '' + (document.body.clientWidth - 160 - 300));
canvas.setAttribute('height', '' + (document.body.clientHeight - 61));

let layers = [];
let r = 15;

function addDenseLayer(name, nodes, activationFunction) {
    layers.push({
        type: 'dense',
        name: name,
        nodes: nodes,
        activation_function: activationFunction
    });

    let canvas = document.querySelector('#canvas');
    let context = canvas.getContext('2d');

    let x = r * 1.2 + (layers.length - 1) * 4 * r;

    for(let i = 0, y = r * 1.2; i < nodes; i++, y += r * 2.5) {
        context.beginPath();
        context.arc(x, y, r, 0, 2 * Math.PI);
        context.stroke();
    }

    if(layers.length > 1) {
        //
    }
}

function updateLayers() {
    let layersSelect = document.querySelector('#layers');
    layersSelect.innerHTML = '';

    let option = document.createElement('option');
    option.value = '-1';
    option.innerText = 'Add Layer';
    layersSelect.appendChild(option);
    for(let i = 0; i < layers.length; i++) {
        option = document.createElement('option');
        option.value = '' + i;
        option.innerText = layers[i]['name'];
        layersSelect.appendChild(option);
    }
}

function updateLayerProperties(index, type) {
    let name = document.querySelector('#layer-name');
    let nodes = document.querySelector('#nodes');
    let activation = document.querySelector('#activation-function');

    if(type === 'dense') {
        name.style['display'] = 'block';
        nodes.style['display'] = 'block';
        activation.style['display'] = 'block';
    }

    if(index === -1) {
        name.value = '';
        nodes.value = '';
        activation.selectedIndex = 0;
    }
}

function clearInputs() {
    let name = document.querySelector('#layer-name');
    let nodes = document.querySelector('#nodes');
    let activation = document.querySelector('#activation-function');

    name.value = '';
    nodes.value = '';
    activation.selectedIndex = 0;
}



// function resizeCanvas() {
//     let canvas = document.querySelector('#canvas');
// }
//
// window.onresize = resizeCanvas;