let canvas = document.querySelector('#canvas');
canvas.setAttribute('width', '' + (document.body.clientWidth - 160 - 300));
canvas.setAttribute('height', '' + (document.body.clientHeight - 61));

let layers = [];
let r = 15;

function addDenseLayer(name, nodes, activation_function) {
    layers.push({
        type: 'dense',
        name: name,
        nodes: nodes,
        activation_function: activation_function
    });

    let canvas = document.querySelector('#canvas');
    let context = canvas.getContext('2d');

    let x = r * 1.2;

    for(let i = 0, y = r * 1.2; i < nodes; i++, y += r * 2.5) {
        context.beginPath();
        context.arc(x, y, r, 0, 2 * Math.PI);
        context.stroke();
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

    if(index === '-1') {
        name.value = '';
        nodes.value = '';
        activation.querySelector('option:first-child').selectedIndex = 0;
    }
}



// function resizeCanvas() {
//     let canvas = document.querySelector('#canvas');
// }
//
// window.onresize = resizeCanvas;