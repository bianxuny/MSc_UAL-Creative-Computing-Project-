const textToWrite = "You say that you love me ! ";
const SEGMENTS = 600;
const RAIN_COUNT = 200; // Number of raindrops
let raindrops = [];

// auto start variables
let centerX, centerY, fontSize, INNER_RADIUS, RADIUS_VARIATION;
let leftEye, rightEye;
let leftFoot, rightFoot;

function setup() {
  createCanvas(windowWidth, windowHeight);
  centerX = windowWidth / 2;
  centerY = windowHeight / 2;

  let screenPct = min(height, width) / 1000;
  fontSize = screenPct * 150;
  INNER_RADIUS = screenPct * 200;
  RADIUS_VARIATION = screenPct * 200;

  textFont("Helvetica");
  textSize(fontSize);

  // Create eyes
  leftEye = new Eye(centerX - 150, centerY - 100, 50);
  rightEye = new Eye(centerX + 150, centerY - 100, 50);

  // Initialize feet
  leftFoot = new Foot(centerX - 100, centerY + 250, 40, -0.1);
  rightFoot = new Foot(centerX + 100, centerY + 250, 40, 0.1);

  // Initialize raindrops
  for (let i = 0; i < RAIN_COUNT; i++) {
    raindrops.push(new Raindrop(random(width), random(-height, 0)));
  }
}

function pointForIndex(pct) {
  const NOISE_SCALE = 1.5;
  let angle = pct * TWO_PI;
  let cosAngle = cos(angle);
  let sinAngle = sin(angle);
  let time = frameCount / 100;
  let noiseValue = noise(
    NOISE_SCALE * cosAngle + NOISE_SCALE,
    NOISE_SCALE * sinAngle + NOISE_SCALE,
    time
  );
  let radius = INNER_RADIUS + RADIUS_VARIATION * noiseValue;
  return {
    x: radius * cosAngle + centerX,
    y: radius * sinAngle + centerY,
  };
}

function draw() {
  background(255, 255, 255);
  noStroke();

  // Draw character
  fill(255, 96, 0); // Yellow color
  beginShape();
  for (let i = 0; i < SEGMENTS; i++) {
    let p0 = pointForIndex(i / SEGMENTS);
    vertex(p0.x, p0.y);
  }
  endShape(CLOSE);

  // Draw text
  let pct = atan2(mouseY - centerY, mouseX - centerX) / TWO_PI; // follow mouse
  let pixToAngularPct = 1 / ((INNER_RADIUS + RADIUS_VARIATION / 2) * TWO_PI);

  fill(255, 161, 0); // Orange color
  for (let i = 0; i < textToWrite.length; i++) {
    let charWidth = textWidth(textToWrite.charAt(i));
    pct += (charWidth / 2) * pixToAngularPct;

    // Calculate angle
    let leftP = pointForIndex(pct - 0.01);
    let rightP = pointForIndex(pct + 0.01);
    let angle = atan2(leftP.y - rightP.y, leftP.x - rightP.x) + PI;

    push();
    let p = pointForIndex(pct);
    translate(p.x, p.y);
    rotate(angle);
    translate(-p.x, -p.y);

    text(textToWrite.charAt(i), p.x - charWidth / 2, p.y);
    pop();

    pct += (charWidth / 2) * pixToAngularPct;
  }

  // Draw eyes
  leftEye.update(mouseX, mouseY);
  rightEye.update(mouseX, mouseY);
  leftEye.display();
  rightEye.display();

  // Draw legs and feet
  leftFoot.update(mouseX, mouseY);
  leftFoot.display(centerX - 100, centerY + 100);

  rightFoot.update(mouseX, mouseY);
  rightFoot.display(centerX + 100, centerY + 100);

  // Draw raindrops
  for (let raindrop of raindrops) {
    raindrop.update();
    raindrop.display();
  }
}

// Eye class to represent eyes that follow the mouse
class Eye {
  constructor(_x, _y, _size) {
    this.x = _x;
    this.y = _y;
    this.size = _size;
    this.pupilSize = _size / 3; // Size of the pupil
    this.pupilX = _x;
    this.pupilY = _y;
  }

  update(mx, my) {
    let angle = atan2(my - this.y, mx - this.x);
    let maxDistance = this.size / 3;
    this.pupilX = this.x + cos(angle) * maxDistance;
    this.pupilY = this.y + sin(angle) * maxDistance;
  }

  display() {
    push();
    translate(this.x, this.y);
    fill(255); // White part of the eye
    noStroke();
    ellipse(0, 0, this.size, this.size);

    fill(0); // Black part of the eye (pupil)
    ellipse(
      this.pupilX - this.x,
      this.pupilY - this.y,
      this.pupilSize,
      this.pupilSize
    );
    pop();
  }
}

// Foot class to represent falling and bouncing feet
class Foot {
  constructor(_x, _y, _size, swingDirection) {
    this.x = _x;
    this.y = _y;
    this.size = _size;
    this.angle = 0; // Initial angle
    this.swingDirection = swingDirection;
    this.swingSpeed = 0.05; // Speed of swinging
  }

  update(mx, my) {
    this.angle =
      sin(frameCount * this.swingSpeed + this.swingDirection) * PI * 0.1;
  }

  display(legX, legY) {
    stroke(0);
    strokeWeight(5);
    line(
      legX,
      legY,
      this.x + cos(this.angle) * 20,
      this.y + sin(this.angle) * 20
    );

    fill(0);
    noStroke();
    ellipse(
      this.x + cos(this.angle) * 20,
      this.y + sin(this.angle) * 20,
      this.size,
      this.size / 2
    );
  }
}

// Raindrop class
class Raindrop {
  constructor(_x, _y) {
    this.x = _x;
    this.y = _y;
    this.size = random(5, 10);
    this.speed = random(2, 5);
  }

  update() {
    this.y += this.speed;

    // Check collision with character
    let d = dist(this.x, this.y, centerX, centerY);
    if (d < INNER_RADIUS + RADIUS_VARIATION / 2) {
      this.y -= this.speed * 2; // Bounce back
    }

    // Reset position if it goes below the screen
    if (this.y > height) {
      this.y = random(-height, 0);
      this.x = random(width);
    }
  }

  display() {
    fill(183, 233, 255); // Blue raindrops
    noStroke();
    ellipse(this.x, this.y, this.size, this.size * 1.5);
  }
}
