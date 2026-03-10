// Main WebGPU script to render an interactive rotating globe.
// Place this file at: WebGPUPlayground/webgpu-globe/main.js
// Usage: include a script tag in an HTML file that creates a canvas with id "gpu-canvas"
// Example:
// <canvas id="gpu-canvas" width="800" height="600"></canvas>
// <script type="module" src="./main.js"></script>

const canvas = document.getElementById("gpu-canvas") || createDemoCanvas();
let device, context, format;

// Public configuration
const EARTH_TEXTURE_URL = "https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular_projection_SW.jpg"; // public domain-ish texture; replace if you want local copy

// ---- Initialization ----
await initWebGPU();
const sphere = createSphere(64, 32, 1.0); // high-res sphere
const resources = createGPUResources(sphere);
const state = createRenderState(resources);
startRenderLoop(state);

// ---- Functions ----

async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("Failed to get GPU adapter");
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });
  canvas.style.touchAction = "none";
}

function createDemoCanvas() {
  const c = document.createElement("canvas");
  c.id = "gpu-canvas";
  c.width = 1024;
  c.height = 640;
  document.body.appendChild(c);
  return c;
}

// ---- Geometry generator: sphere ----
function createSphere(longitudes = 32, latitudes = 16, radius = 1.0) {
  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];

  for (let y = 0; y <= latitudes; y++) {
    const v = y / latitudes;
    const theta = v * Math.PI; // 0..PI

    for (let x = 0; x <= longitudes; x++) {
      const u = x / longitudes;
      const phi = u * Math.PI * 2; // 0..2PI

      const sinTheta = Math.sin(theta);
      const pos = [
        radius * Math.cos(phi) * sinTheta,
        radius * Math.cos(theta),
        radius * Math.sin(phi) * sinTheta,
      ];
      positions.push(...pos);

      const n = normalize(pos);
      normals.push(...n);

      // equirectangular uv
      uvs.push(u, 1 - v);
    }
  }

  for (let y = 0; y < latitudes; y++) {
    for (let x = 0; x < longitudes; x++) {
      const i1 = y * (longitudes + 1) + x;
      const i2 = i1 + longitudes + 1;

      indices.push(i1, i2, i1 + 1);
      indices.push(i1 + 1, i2, i2 + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: indices.length > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  };
}

function normalize(v) {
  const len = Math.hypot(...v);
  if (len === 0) return [0, 0, 0];
  return v.map((x) => x / len);
}

// ---- GPU resource creation ----
function createGPUBuffer(arr, usage) {
  const [typedArray, size] = arr instanceof ArrayBuffer ? [new Uint8Array(arr), arr.byteLength] : [arr, arr.byteLength];
  const buffer = device.createBuffer({
    size: align(size, 4),
    usage,
    mappedAtCreation: true,
  });
  const mapping = new (typedArray.constructor)(buffer.getMappedRange());
  mapping.set(typedArray);
  buffer.unmap();
  return buffer;
}

function align(n, alignment) {
  return Math.ceil(n / alignment) * alignment;
}

function createGPUResources(sphere) {
  const positionBuffer = createGPUBuffer(sphere.positions, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  const normalBuffer = createGPUBuffer(sphere.normals, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  const uvBuffer = createGPUBuffer(sphere.uvs, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  const indexBuffer = createGPUBuffer(sphere.indices, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);

  const uniformBufferSize = 4 * 16 + 4 * 16 + 4 * 4; // mvp(4x4) + model(4x4) + vec3(light) padded
  const uniformBuffer = device.createBuffer({
    size: align(uniformBufferSize, 256),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // create placeholder 1x1 blue texture immediately to avoid render glitch
  const placeholderTex = device.createTexture({
    size: [1, 1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.writeTexture(
    { texture: placeholderTex },
    new Uint8Array([100, 149, 237, 255]),
    { bytesPerRow: 4 },
    [1, 1, 1]
  );

  // Try load earth image asynchronously and produce a GPUTexture; start load in background
  const texturePromise = loadImage(EARTH_TEXTURE_URL)
    .then((img) => createTextureFromImage(img))
    .catch((err) => {
      console.warn("Failed to load texture:", err);
      // fallback: generate checkerboard canvas
      const c = genCheckerboardCanvas(1024, 512);
      return createTextureFromImage(c);
    });

  return {
    positionBuffer,
    normalBuffer,
    uvBuffer,
    indexBuffer,
    indexCount: sphere.indices.length,
    uniformBuffer,
    sampler,
    texturePromise,
    texture: placeholderTex,
  };
}

function genCheckerboardCanvas(w = 1024, h = 512) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  const cols = 32;
  const rows = 16;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? "#88a" : "#dde";
      ctx.fillRect((x * w) / cols, (y * h) / rows, w / cols, h / rows);
    }
  }
  // draw simple meridians/latitude lines for recognition
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  for (let i = 0; i < cols; i++) {
    const u = (i / cols) * w;
    ctx.beginPath();
    ctx.moveTo(u, 0);
    ctx.lineTo(u, h);
    ctx.stroke();
  }
  return c;
}

async function createTextureFromImage(imageSource) {
  // imageSource can be HTMLImageElement, HTMLCanvasElement, or ImageBitmap
  const imgBitmap = imageSource instanceof ImageBitmap ? imageSource : await createImageBitmap(imageSource);
  const tex = device.createTexture({
    size: [imgBitmap.width, imgBitmap.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: imgBitmap },
    { texture: tex },
    [imgBitmap.width, imgBitmap.height]
  );
  return tex;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

// ---- Shaders (WGSL strings) ----
const shaderWGSL = `struct Uniforms {
  mvpMatrix : mat4x4<f32>;
  modelMatrix : mat4x4<f32>;
  lightDir : vec4<f32>;
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var earthSampler : sampler;
@group(0) @binding(2) var earthTexture : texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3<f32>;
  @location(1) normal : vec3<f32>;
  @location(2) uv : vec2<f32>;
};

struct VertexOutput {
  @builtin(position) Position : vec4<f32>;
  @location(0) vNormal : vec3<f32>;
  @location(1) vUV : vec2<f32>;
  @location(2) vWorldPos : vec3<f32>;
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var out : VertexOutput;
  let worldPos = (uniforms.modelMatrix * vec4<f32>(input.position, 1.0)).xyz;
  out.Position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
  out.vNormal = normalize((uniforms.modelMatrix * vec4<f32>(input.normal, 0.0)).xyz);
  out.vUV = input.uv;
  out.vWorldPos = worldPos;
  return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  let albedo = textureSample(earthTexture, earthSampler, in.vUV).rgb;
  // simple Lambert + ambient
  let n = normalize(in.vNormal);
  let l = normalize(uniforms.lightDir.xyz);
  let diff = max(dot(n, l), 0.0);
  let ambient = 0.2;
  let color = albedo * (ambient + 0.8 * diff);
  return vec4<f32>(color, 1.0);
}
`;

// ---- Pipeline and render state ----
function createRenderState(resources) {
  // Create shader module
  const module = device.createShaderModule({ code: shaderWGSL });

  const vertexBuffers = [
    {
      arrayStride: 4 * 3,
      attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
    },
    {
      arrayStride: 4 * 3,
      attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
    },
    {
      arrayStride: 4 * 2,
      attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }],
    },
  ];

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: vertexBuffers,
    },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  // depth texture
  let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // bind group layout and bind group (auto layout used but create bindGroup)
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.uniformBuffer } },
      { binding: 1, resource: resources.sampler },
      { binding: 2, resource: resources.texture.createView ? resources.texture.createView() : resources.textureView },
    ],
  });

  // pointer control state
  const control = createCameraController();

  // attempt to replace placeholder texture when actual texture loads
  resources.texturePromise.then((tex) => {
    resources.texture = tex;
    // recreate bind group with the real texture
    const newBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: resources.uniformBuffer } },
        { binding: 1, resource: resources.sampler },
        { binding: 2, resource: tex.createView() },
      ],
    });
    state.bindGroup = newBindGroup;
  }).catch(() => { /* handled earlier */ });

  const state = {
    pipeline,
    resources,
    bindGroup,
    depthTexture,
    control,
    rotationX: 0,
    rotationY: 0,
    zoom: 3.5,
    lastTimestamp: 0,
    animating: true,
  };

  // override bindGroup after resources ready
  state.bindGroup = bindGroup;

  // handle resize
  const ro = new ResizeObserver(() => resizeSwapChainAndDepth(state));
  ro.observe(canvas);

  // initialize uniforms
  updateUniforms(state);

  // attach input events
  attachInteraction(canvas, state);

  return state;
}

function resizeSwapChainAndDepth(state) {
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio));
  state.depthTexture.destroy();
  state.depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

// ---- Camera/controller ----
function createCameraController() {
  return {
    dragging: false,
    lastX: 0,
    lastY: 0,
  };
}

function attachInteraction(canvas, state) {
  const ctrl = state.control;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    ctrl.dragging = true;
    ctrl.lastX = e.clientX;
    ctrl.lastY = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => {
    ctrl.dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!ctrl.dragging) return;
    const dx = e.clientX - ctrl.lastX;
    const dy = e.clientY - ctrl.lastY;
    ctrl.lastX = e.clientX;
    ctrl.lastY = e.clientY;
    const k = 0.005;
    state.rotationY += dx * k;
    state.rotationX += dy * k;
    state.rotationX = clamp(state.rotationX, -Math.PI / 2, Math.PI / 2);
    updateUniforms(state);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const dy = e.deltaY;
    state.zoom *= 1 + dy * 0.001;
    state.zoom = clamp(state.zoom, 1.6, 8.0);
    updateUniforms(state);
  }, { passive: false });
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

// ---- Math helpers ----
function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; ++i) {
    for (let j = 0; j < 4; ++j) {
      let sum = 0;
      for (let k = 0; k < 4; ++k) {
        sum += a[i * 4 + k] * b[k * 4 + j];
      }
      out[i * 4 + j] = sum;
    }
  }
  return out;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = f;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[12] = 0;
  out[13] = 0;
  out[14] = (2 * far * near) * nf;
  out[15] = 0;
  return out;
}

function mat4Translation(tx, ty, tz) {
  const out = mat4Identity();
  out[12] = tx;
  out[13] = ty;
  out[14] = tz;
  return out;
}

function mat4RotationX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  const out = mat4Identity();
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  return out;
}

function mat4RotationY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  const out = mat4Identity();
  out[0] = c;
  out[2] = -s;
  out[8] = s;
  out[10] = c;
  return out;
}

function mat4Scale(sx, sy, sz) {
  const out = mat4Identity();
  out[0] = sx;
  out[5] = sy;
  out[10] = sz;
  return out;
}

// ---- Uniform updates ----
function updateUniforms(state) {
  // Build projection * view * model
  const aspect = canvas.width / canvas.height;
  const proj = mat4Perspective((60 * Math.PI) / 180, aspect, 0.1, 100.0);
  const camPos = [0, 0, state.zoom];
  const view = mat4Translation(-camPos[0], -camPos[1], -camPos[2]);
  const rotY = mat4RotationY(state.rotationY);
  const rotX = mat4RotationX(state.rotationX);
  const model = mat4Multiply(rotY, rotX);
  const mvp = mat4Multiply(proj, mat4Multiply(view, model));

  // normalised light direction in world space
  const lightDir = normalize([0.5, 0.8, 0.6]);

  // pack into buffer (mvp, model, lightDir padded)
  const uniformData = new Float32Array(16 + 16 + 4);
  uniformData.set(mvp, 0);
  uniformData.set(model, 16);
  uniformData.set([...lightDir, 0.0], 32);

  device.queue.writeBuffer(state.resources.uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset, uniformData.byteLength);
}

// ---- Render loop ----
function startRenderLoop(state) {
  function frame(ts) {
    if (!state.lastTimestamp) state.lastTimestamp = ts;
    const dt = (ts - state.lastTimestamp) / 1000;
    state.lastTimestamp = ts;

    // automatic slow spin when not dragging
    if (!state.control.dragging) {
      state.rotationY += dt * 0.25; // 0.25 rad/s
      updateUniforms(state);
    }

    // replace bindGroup's texture if texturePromise resolved with real texture
    if (state.resources.texturePromise && state.resources.texture) {
      // ensure bind group updated earlier in createRenderState when promise resolves
    }

    render(state);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function render(state) {
  // ensure canvas sized properly
  const devicePixelRatio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.floor(canvas.clientWidth * devicePixelRatio) ||
      canvas.height !== Math.floor(canvas.clientHeight * devicePixelRatio)) {
    canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
    canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
    state.depthTexture.destroy();
    state.depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();

  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0.05, g: 0.08, b: 0.13, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
    depthStencilAttachment: {
      view: state.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(state.pipeline);
  // bind buffers
  renderPass.setVertexBuffer(0, state.resources.positionBuffer);
  renderPass.setVertexBuffer(1, state.resources.normalBuffer);
  renderPass.setVertexBuffer(2, state.resources.uvBuffer);
  renderPass.setIndexBuffer(state.resources.indexBuffer, state.resources.indexCount > 65535 ? "uint32" : "uint16");
  renderPass.setBindGroup(0, state.bindGroup);
  renderPass.drawIndexed(state.resources.indexCount, 1, 0, 0, 0);
  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);
}

// ---- Utilities ----
// Simple helper: create a canvas if none provided for demo
// (already called at top when canvas missing)
function hexToRgba(hex) {
  const m = hex.replace("#", "");
  const bigint = parseInt(m, 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255, 255];
}
