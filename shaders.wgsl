// WGSL shaders for a simple WebGPU globe renderer.
//
// - Vertex shader transforms sphere vertices into clip space and passes
//   world-space normal, UV and position to the fragment shader.
// - Fragment shader samples an Earth texture, applies diffuse + specular
//   lighting, and adds a subtle atmospheric rim glow.
//
// Bindings:
// @group(0) @binding(0) : uniform buffer with matrices and light/time
// @group(0) @binding(1) : sampler
// @group(0) @binding(2) : 2D texture (earth color map)
//
// Notes for the host:
// - Provide a model-view-projection matrix in `uniforms.mvp`.
// - Provide a `uniforms.model` matrix (model -> world). If camera is in world
//   origin, view direction is approximated as -worldPosition.
// - `uniforms.lightDirTime.xyz` holds a normalized light direction (pointing
//   from the surface toward the light). `lightDirTime.w` can be used for time.

struct Uniforms {
    mvp: mat4x4<f32>;
    model: mat4x4<f32>;
    // lightDirection.xyz is the light direction (should be normalized).
    // lightDirection.w can optionally carry time or other parameter.
    lightDirTime: vec4<f32>;
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(0) @binding(1) var u_sampler: sampler;
@group(0) @binding(2) var u_texture: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>;
    @location(1) normal: vec3<f32>;
    @location(2) uv: vec2<f32>;
};

struct VertexOutput {
    @builtin(position) Position: vec4<f32>;
    @location(0) vUV: vec2<f32>;
    @location(1) vNormal: vec3<f32>;
    @location(2) vWorldPos: vec3<f32>;
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform position to clip space
    out.Position = uniforms.mvp * vec4<f32>(input.position, 1.0);

    // Transform to world-space position and normal
    let worldPos4 = uniforms.model * vec4<f32>(input.position, 1.0);
    out.vWorldPos = worldPos4.xyz;

    // For normals, use the model matrix's linear part; if model contains
    // non-uniform scale a proper inverse-transpose should be used.
    // Here we assume either uniform scale or accept slight inaccuracy.
    let worldNormal4 = uniforms.model * vec4<f32>(input.normal, 0.0);
    out.vNormal = normalize(worldNormal4.xyz);

    out.vUV = input.uv;

    return out;
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    // Schlick approximation for fresnel
    return F0 + (vec3<f32>(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample diffuse color from the texture (assume sRGB -> linear already handled by host)
    let albedo = textureSample(u_texture, u_sampler, in.vUV).rgb;

    // World-space normal (already normalized in vertex shader, but ensure it here)
    let N = normalize(in.vNormal);

    // Light direction: uniforms.lightDirTime.xyz expected to be normalized and point TO the light
    let L = normalize(uniforms.lightDirTime.xyz);

    // View direction: assume camera at world origin (0,0,0)
    let V = normalize(-in.vWorldPos);

    // Diffuse Lambertian
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = albedo * NdotL;

    // Simple Blinn-Phong specular
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);
    let shininess = 64.0;
    let specularFactor = pow(NdotH, shininess);

    // Use a modest specular color; take a small metallic-ish F0 from albedo luminance
    let luminance = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
    let F0 = mix(vec3<f32>(0.02, 0.02, 0.02), albedo, 0.05); // small blend
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let specular = F * specularFactor * 0.8;

    // Ambient term (subtle)
    let ambient = albedo * 0.12;

    // Rim/atmosphere glow: stronger when viewing edge-on
    let rim = pow(max(1.0 - dot(N, V), 0.0), 2.0);
    let atmosphereColor = vec3<f32>(0.4, 0.6, 1.0) * 0.35 * rim;

    // Combine
    var color = ambient + diffuse * 0.98 + specular + atmosphereColor;

    // Tone mapping + gamma correction (ACES-like simple approx)
    color = color / (color + vec3<f32>(1.0, 1.0, 1.0));
    // Convert to sRGB gamma
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
