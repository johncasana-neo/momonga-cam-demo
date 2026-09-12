import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
const GESTURE_MEMES = {
  default: ["memes/momonga_default.jpg"],
  huhCat: ["memes/momonga_huhcat.jpg"],
  sideEyeCat: ["memes/momonga_side.jpg"],
  sideEyeDownCat: ["memes/momonga_down.jpg"],
  fist: ["memes/momonga_fist.jpg"],
};

// how many consecutive frames a gesture must hold before we switch to it
const STABLE_FRAMES_REQUIRED = 5;
// if no hand / no gesture is seen for this long, fall back to default
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
// (e.g. hand covering the mouth during a shush)
const FACE_STALE_MS = 1200;

// how far the head has to turn (yaw, in degrees, from MediaPipe's own head
// pose estimate - not a hand-rolled distance heuristic) to count as a
// side-eye look. Watch the live debug HUD in the camera pane while turning
// your head to find the right value for you.
const SIDE_EYE_YAW_DEG = 15.0;

// same idea, but for tilting the head DOWN (pitch) instead of turning it
// sideways (yaw) - "slight side eye" when you look down.
const SIDE_EYE_DOWN_PITCH_DEG = 12.0;

// huh cat: mouth open AND eyes wide, via MediaPipe's face blendshapes
// (jawOpen / eyeWideLeft / eyeWideRight) rather than hand-rolled geometry.
const HUH_JAW_THRESHOLD = 0.03;
const EYE_WIDE_THRESHOLD = 0.01;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, yawDeg, t }
let lastYawDebug = 0;
let lastPitchDebug = 0;
let lastJawOpen = 0;
let lastEyeWide = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
// Using z (depth) as well as x/y makes these tests far more robust to hand
// rotation, foreshortening, and motion blur than a plain 2D/wrist-distance
// check would be.
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

// extract the head's left/right turn angle (yaw, degrees) from MediaPipe's
// facial transformation matrix - its own estimate of head pose, far more
// robust than trying to infer turn from landmark distances.
function yawFromTransformMatrix(matrixData) {
  // matrixData is a 16-element row-major 4x4 array; r(row, col) = data[row*4+col]
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

function classifyHand(lm) {
  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);
  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;
  return { curledCount };
}

// extract the head's up/down tilt angle (pitch, degrees) from the same
// transformation matrix as yaw above.
function pitchFromTransformMatrix(matrixData) {
  const r21 = matrixData[9];
  const r22 = matrixData[10];
  return (Math.atan2(r21, r22) * 180) / Math.PI;
}

// pull MediaPipe's face blendshapes into a plain {name: score} dict.
function blendshapeScores(faceResult) {
  if (!faceResult.faceBlendshapes || faceResult.faceBlendshapes.length === 0) return {};
  const out = {};
  for (const b of faceResult.faceBlendshapes[0].categories) out[b.categoryName] = b.score;
  return out;
}

function eyeWideScore(scores) {
  return Math.max(scores.eyeWideLeft || 0, scores.eyeWideRight || 0);
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    // how open the mouth is right now - normalized so it doesn't depend on
    // distance from the camera.
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    let yawDeg = 0;
    let pitchDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const matrix = faceResult.facialTransformationMatrixes[0].data;
      yawDeg = yawFromTransformMatrix(matrix);
      pitchDeg = pitchFromTransformMatrix(matrix);
    }

    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, t: now };
    lastYawDebug = yawDeg;
    lastPitchDebug = pitchDeg;

    const scores = blendshapeScores(faceResult);
    lastJawOpen = scores.jawOpen || 0;
    lastEyeWide = eyeWideScore(scores);
  }
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    // no hands: huh and side-eye are both face-only poses.
    if (faceIsFresh && lastJawOpen > HUH_JAW_THRESHOLD && lastEyeWide > EYE_WIDE_THRESHOLD) {
      return "huhCat";
    }
    if (faceIsFresh && Math.abs(lastFace.yawDeg) > SIDE_EYE_YAW_DEG) {
      return "sideEyeCat";
    }
    if (faceIsFresh && lastPitchDebug > SIDE_EYE_DOWN_PITCH_DEG) {
      return "sideEyeDownCat";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  if (hands.some((h) => h.curledCount === 4)) {
    return "fist";
  }

  // hand up but not a fist - still allow a strong side-eye read to win
  // over an ambiguous hand pose.
  if (faceIsFresh && Math.abs(lastFace.yawDeg) > SIDE_EYE_YAW_DEG) {
    return "sideEyeCat";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // debounce: require a gesture to be seen for several consecutive
    // frames before we commit to it, to avoid flicker between frames
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `gesture: ${currentGesture}\n` +
    `yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} deg  (side-eye thr +/-${SIDE_EYE_YAW_DEG.toFixed(1)})\n` +
    `pitch: ${lastPitchDebug >= 0 ? "+" : ""}${lastPitchDebug.toFixed(1)} deg  (side-eye-down thr ${SIDE_EYE_DOWN_PITCH_DEG.toFixed(1)})\n` +
    `jawOpen: ${lastJawOpen.toFixed(2)}  eyeWide: ${lastEyeWide.toFixed(2)}  (huh needs both > ${HUH_JAW_THRESHOLD.toFixed(2)}/${EYE_WIDE_THRESHOLD.toFixed(2)})`;
}

init().catch((err) => console.error(err));
