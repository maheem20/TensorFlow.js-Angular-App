import { Component, ViewChild, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent implements OnInit {
  title = 'tensorApp';

  linearModel: tf.Sequential | undefined;
  prediction: any;

  ngOnInit() {
    this.trainNewModel();
  }

  async trainNewModel() {
    // Define a model for linear regression.
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    // Prepare the model for training: Specify the loss and the optimizer.
    this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Training data, completely random stuff
    const xs = tf.tensor1d([3.2, 4.4, 5.5, 6.71, 6.98, 7.168,
      9.779, 6.182, 7.59, 2.167, 7.042, 10.791,
      5.313, 7.997, 5.654, 9.27, 3.1]);
    const ys = tf.tensor1d([1.6, 2.7, 2.9, 3.19, 1.684, 2.53,
      3.366, 2.596, 2.53, 1.221, 2.827, 3.465,
      1.65, 2.904, 2.42, 2.94, 1.3]);
    // Train
    await this.linearModel.fit(xs, ys);
    console.log('model trained!');
  }

  linearPrediction(val: number) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }
}