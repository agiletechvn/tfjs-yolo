import yolo from '../tfjs-yolo';
import * as tf from '@tensorflow/tfjs';

console.log(tf.getBackend());

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fileinput = document.getElementById('image_input');

let myYolo;

(async function main() {
  try {
    myYolo = await yolo.v3tiny();
    const boxes = await myYolo.predict(new Image('./dist/person.jpg'));
    // console.log(myYolo);
    fileinput.style.display = 'block';
    fileinput.onchange = function(event) {
      console.log(event);
      var img = new Image();
      img.onload = async function() {
        canvas.height = img.height;
        canvas.width = img.width;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        await run();
      };
      img.src = URL.createObjectURL(event.target.files[0]);
    };
  } catch (e) {
    console.error(e);
  }
})();

async function run() {
  console.log('Start with tensors: ' + tf.memory().numTensors);
  const start = Date.now();
  const boxes = await myYolo.predict(canvas);
  const collapsed = Date.now() - start + ' ms';
  console.log(collapsed, boxes);
  boxes.map(box => {
    ctx.lineWidth = 2;
    ctx.fillStyle = 'red';
    ctx.strokeStyle = 'red';
    ctx.rect(box['left'], box['top'], box['width'], box['height']);
    ctx.fillText(box['class'], box['left'] + 5, box['top'] + 10);
    ctx.fillText(collapsed, 5, 10);

    ctx.stroke();
  });
  console.log('End with tensors: ' + tf.memory().numTensors);
}
