// Split Radiance Cascades native compute graph.
// Generated Rust bindings are committed in src/shader.rs.

@block types
#define EMPTY 0xFFFFFFFFu
#define SLOT_TOMBSTONE 0xFFFFFFFCu
#define INDEX_RESERVED 0xFFFFFFFEu
#define INDEX_DUPLICATE 0xFFFFFFFDu
#define LEAF_BIT 0x80000000u
#define PI 3.14159265358979323846
#define TAU 6.28318530717958647692
#define FIXED_SCALE 4096.0
#define MAX_RADIANCE 16.0
#define BVH_STACK 64
#define RANK_BLOCK_SIZE 512u

const uint PROBE_CAPS[4] = uint[4](4096u, 1536u, 512u, 256u);
const uint HASH_CAPS[4] = uint[4](8192u, 4096u, 2048u, 1024u);
const uint PROBE_OFFSETS[4] = uint[4](0u, 4096u, 5632u, 6144u);
const uint HASH_OFFSETS[4] = uint[4](0u, 8192u, 12288u, 14336u);
const uint DIR_COUNTS[4] = uint[4](32u, 128u, 512u, 2048u);
const uint DATA_OFFSETS[4] = uint[4](0u, 131072u, 327680u, 589824u);
const uint TOTAL_PROBES = 6400u;
const uint TOTAL_HASH = 15360u;
const uint TOTAL_VALUES = 1114112u;

struct bvh_node {
    vec3 bounds_min;
    uint left_or_first;
    vec3 bounds_max;
    uint right_or_count;
};

struct triangle {
    vec4 a;
    vec4 edge_ab;
    vec4 edge_ac;
    vec4 albedo;
    vec4 emissive;
    vec4 uv_ab;
    vec4 uv_c_material;
    uvec4 normal_oct;
};

struct surface {
    vec4 position;
    vec4 normal_depth;
    vec4 albedo;
    vec4 direct_emissive;
};

struct ray_hit {
    float t;
    vec3 normal;
    vec3 albedo;
    vec3 emissive;
    uint triangle_id;
};

struct word {
    uint value;
};

uint hash32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

vec2 r2(uint index, vec2 jitter) {
    const float g = 1.324717957244746;
    return fract(vec2(0.5) + vec2(float(index) / g, float(index) / (g * g)) + jitter);
}

vec3 decode_equal_area(vec2 uv) {
    float phi = uv.x * TAU;
    float z = uv.y * 2.0 - 1.0;
    float radius = sqrt(max(0.0, 1.0 - z * z));
    return vec3(radius * cos(phi), radius * sin(phi), z);
}

vec2 encode_equal_area(vec3 direction) {
    direction = normalize(direction);
    return vec2(fract(atan(direction.y, direction.x) / TAU + 1.0), direction.z * 0.5 + 0.5);
}

vec2 oct_encode(vec3 n) {
    n /= max(1.0e-8, abs(n.x) + abs(n.y) + abs(n.z));
    vec2 p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * sign(p.xy + vec2(1.0e-8));
    }
    return p * 0.5 + 0.5;
}

vec3 oct_decode(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * sign(n.xy + vec2(1.0e-8));
    }
    return normalize(n);
}

bool key_cell_valid(ivec3 cell, uint lod) {
    return !any(lessThan(cell, ivec3(-262144))) &&
           !any(greaterThan(cell, ivec3(262143))) &&
           lod <= 7u;
}

uint pack_key_low(ivec3 cell, uint lod) {
    if (!key_cell_valid(cell, lod)) return 0u;
    uvec3 c = uvec3(cell + ivec3(262144));
    return c.x | ((c.y & 0x1fffu) << 19u);
}

uint pack_key_high(ivec3 cell, uint lod, bool secondary) {
    if (!key_cell_valid(cell, lod)) return EMPTY;
    uvec3 c = uvec3(cell + ivec3(262144));
    return (c.y >> 13u) | (c.z << 6u) | ((lod & 7u) << 25u) |
           (secondary ? (1u << 28u) : 0u);
}

bool key_valid(uint high) {
    return high != EMPTY;
}

bool key_less(uint left_low, uint left_high, uint right_low, uint right_high) {
    return left_high < right_high || (left_high == right_high && left_low < right_low);
}

uint key_hash(uint low, uint high) {
    return hash32(low ^ hash32(high)) & 0x7fffffffu;
}

ivec3 probe_cell(vec3 position, uint cascade, uint lod, float base_spacing) {
    float spacing = base_spacing * exp2(float(cascade + lod));
    return ivec3(floor(position / spacing));
}

vec3 probe_position(ivec3 cell, uint cascade, uint lod, float base_spacing) {
    float spacing = base_spacing * exp2(float(cascade + lod));
    return (vec3(cell) + vec3(0.5)) * spacing;
}

uint morton_direction_index(uint u, uint v, uint cascade) {
    uint bits = 2u + cascade;
    uint result = 0u;
    for (uint bit = 0u; bit < bits; bit++) {
        result |= ((u >> bit) & 1u) << (bit * 2u);
        result |= ((v >> bit) & 1u) << (bit * 2u + 1u);
    }
    return result | (((u >> bits) & 1u) << (bits * 2u));
}

uvec2 morton_direction_coordinates(uint index, uint cascade) {
    uint bits = 2u + cascade;
    uvec2 result = uvec2(0u);
    for (uint bit = 0u; bit < bits; bit++) {
        result.x |= ((index >> (bit * 2u)) & 1u) << bit;
        result.y |= ((index >> (bit * 2u + 1u)) & 1u) << bit;
    }
    result.x |= ((index >> (bits * 2u)) & 1u) << bits;
    return result;
}

uint direction_index(vec3 direction, uint cascade) {
    uint theta = 4u << cascade;
    vec2 uv = encode_equal_area(direction);
    uint x = min(theta * 2u - 1u, uint(floor(uv.x * float(theta * 2u))));
    uint y = min(theta - 1u, uint(floor(uv.y * float(theta))));
    return morton_direction_index(x, y, cascade);
}

vec3 direction_center(uint index, uint cascade) {
    uint theta = 4u << cascade;
    uint width = theta * 2u;
    uvec2 coordinate = morton_direction_coordinates(index, cascade);
    return decode_equal_area(vec2(
        (float(coordinate.x) + 0.5) / float(width),
        (float(coordinate.y) + 0.5) / float(theta)
    ));
}

float interval_cutoff(float base_spacing, uint cascade, uint lod) {
    return 1.6 * base_spacing * exp2(float(lod)) * pow(4.0, float(cascade));
}

uint data_index(uint cascade, uint probe, uint direction) {
    return DATA_OFFSETS[cascade] + probe * DIR_COUNTS[cascade] + direction;
}

uint fixed_value(float value) {
    return uint(round(clamp(value, 0.0, MAX_RADIANCE) * FIXED_SCALE));
}
@end

@block uniforms
layout(binding=0) uniform frame_params {
    mat4 inverse_view_projection;
    mat4 view_projection;
    mat4 previous_view_projection;
    vec4 camera_time;
    vec4 environment_base_spacing;
    vec4 sun_direction_intensity;
    vec4 point_position_intensity;
    vec4 point_color_exposure;
    ivec4 dimensions_frame_i;
    ivec4 state_layout_0_i;
    ivec4 state_layout_1_i;
    ivec4 state_layout_2_i;
    ivec4 radiance_layout_0_i;
    ivec4 radiance_layout_1_i;
    ivec4 pass_params_i;
    ivec4 feature_flags_i;
};
#define dimensions_frame uvec4(dimensions_frame_i)
#define state_layout_0 uvec4(state_layout_0_i)
#define state_layout_1 uvec4(state_layout_1_i)
#define state_layout_2 uvec4(state_layout_2_i)
#define radiance_layout_0 uvec4(radiance_layout_0_i)
#define radiance_layout_1 uvec4(radiance_layout_1_i)
#define pass_params uvec4(pass_params_i)
#define feature_flags uvec4(feature_flags_i)
@end

@block state_helpers
uint current_frame() { return dimensions_frame.z & 1u; }
uint previous_frame() { return 1u - current_frame(); }
uint sample_count() { return state_layout_2.x; }
uint ray_slot_count() { return state_layout_2.y; }
uint total_state_words() { return state_layout_2.z; }
uint total_radiance_words() { return state_layout_2.w; }

uint counter_active(uint cascade) { return state_layout_1.x + cascade; }
uint counter_ray_count(uint cascade, uint probe) {
    return state_layout_1.x + 4u + PROBE_OFFSETS[cascade] + probe;
}
uint counter_ray_offset(uint cascade, uint probe) {
    return state_layout_1.x + 4u + TOTAL_PROBES + PROBE_OFFSETS[cascade] + probe;
}
uint diagnostics_offset() { return state_layout_1.x + 4u + TOTAL_PROBES * 2u; }
uint retry_counter(uint queue) { return diagnostics_offset() + 4u + queue; }
uint retry_base(uint queue) { return state_layout_1.z + queue * ray_slot_count(); }
void enqueue_retry(uint queue, uint source) {
    uint index = atomicAdd(state_words[retry_counter(queue)].value, 1u);
    if (index < ray_slot_count()) {
        state_words[retry_base(queue) + index].value = source;
    } else {
        atomicAdd(state_words[diagnostics_offset()].value, 1u);
    }
}

uint hash_base(uint cascade, uint frame) {
    return state_layout_0.x + frame * state_layout_0.y + HASH_OFFSETS[cascade] * 5u;
}
uint meta_base(uint cascade, uint frame, uint probe) {
    return state_layout_0.z + frame * state_layout_0.w + (PROBE_OFFSETS[cascade] + probe) * 4u;
}
uint ray_map_base(uint slot) { return state_layout_1.y + slot * 2u; }
uint block_base(uint probe, uint block) {
    return state_layout_1.z + probe * state_layout_1.w + block;
}
uint hit_record_base(uint frame, uint sample_id) {
    return state_layout_1.y + ray_slot_count() * 2u +
           frame * sample_count() * 8u + sample_id * 8u;
}
uint presentation_history_base(uint frame, uint pixel) {
    return radiance_layout_1.z + frame * radiance_layout_1.w + pixel * 16u;
}

vec4 load_state_vec4(uint base) {
    return vec4(
        uintBitsToFloat(state_words[base].value),
        uintBitsToFloat(state_words[base + 1u].value),
        uintBitsToFloat(state_words[base + 2u].value),
        uintBitsToFloat(state_words[base + 3u].value)
    );
}
void store_state_vec4(uint base, vec4 value) {
    state_words[base].value = floatBitsToUint(value.x);
    state_words[base + 1u].value = floatBitsToUint(value.y);
    state_words[base + 2u].value = floatBitsToUint(value.z);
    state_words[base + 3u].value = floatBitsToUint(value.w);
}

uint lookup_probe_frame(uint cascade, uint key_low, uint key_high, uint frame) {
    if (!key_valid(key_high)) return EMPTY;
    uint mask = HASH_CAPS[cascade] - 1u;
    uint token = key_hash(key_low, key_high);
    uint signature = token == EMPTY ? INDEX_RESERVED : token;
    uint start = token & mask;
    uint base = hash_base(cascade, frame);
    for (uint attempt = 0u; attempt < 32u; attempt++) {
        uint slot = (start + attempt) & mask;
        uint address = base + slot * 5u;
        uint stored_signature = state_words[address].value;
        if (stored_signature == EMPTY) {
            return EMPTY;
        }
        if (stored_signature == SLOT_TOMBSTONE) continue;
        uint index = state_words[address + 4u].value;
        if (stored_signature == signature &&
            index != EMPTY &&
            state_words[address + 1u].value == token) {
            uint stored_low = state_words[address + 2u].value;
            uint stored_high = state_words[address + 3u].value;
            if (stored_low == key_low && stored_high == key_high) {
                return index < PROBE_CAPS[cascade] ? index : EMPTY;
            }
        }
    }
    return EMPTY;
}

uint lookup_probe(uint cascade, uint key_low, uint key_high) {
    return lookup_probe_frame(cascade, key_low, key_high, current_frame());
}

void insert_key(uint cascade, uint key_low, uint key_high, out uint deferred) {
    deferred = 0u;
    if (!key_valid(key_high)) {
        atomicAdd(state_words[diagnostics_offset() + 2u].value, 1u);
        return;
    }
    uint mask = HASH_CAPS[cascade] - 1u;
    uint token = key_hash(key_low, key_high);
    // Reserve using a token-derived non-empty signature. The compact-index
    // word is the publication state: EMPTY while claimed, INDEX_RESERVED
    // only after the complete token/full-key tuple is globally visible.
    uint signature = token == EMPTY ? INDEX_RESERVED : token;
    uint start = token & mask;
    uint base = hash_base(cascade, current_frame());
    bool resolved = false;
    for (uint attempt = 0u; attempt < 32u; attempt++) {
        uint slot = (start + attempt) & mask;
        uint address = base + slot * 5u;
        uint old = atomicCompSwap(
            state_words[address].value,
            EMPTY,
            signature
        );
        bool inserted = old == EMPTY;
        if (inserted) {
            atomicExchange(state_words[address + 1u].value, token);
            atomicExchange(state_words[address + 2u].value, key_low);
            atomicExchange(state_words[address + 3u].value, key_high);
            memoryBarrierBuffer();
            atomicExchange(state_words[address + 4u].value, INDEX_RESERVED);
        }
        // Do not return from the winning branch until reconvergence: peers in
        // the same SIMD wave then observe the complete publication.
        if (inserted) {
            resolved = true;
            break;
        }
        if (old == signature) {
            uint observed_index = atomicAdd(
                state_words[address + 4u].value,
                0u
            );
            if (observed_index == EMPTY) {
                // Never probe past an unpublished reservation. The frame graph
                // repeats collection after a dispatch boundary, so a genuine
                // same-token/different-key collision is retried only after the
                // exact full key becomes visible. This cannot create a second
                // slot for an identical key and cannot deadlock a SIMD wave.
                deferred = 1u;
                resolved = true;
                break;
            }
            memoryBarrierBuffer();
            uint published_token = atomicAdd(
                state_words[address + 1u].value,
                0u
            );
            uint published_low = atomicAdd(
                state_words[address + 2u].value,
                0u
            );
            uint published_high = atomicAdd(
                state_words[address + 3u].value,
                0u
            );
            if (published_token == token &&
                published_low == key_low &&
                published_high == key_high) {
                resolved = true;
                break;
            }
            if (published_token == token) {
                atomicAdd(state_words[diagnostics_offset() + 1u].value, 1u);
            }
        }
    }
    if (!resolved) {
        atomicAdd(state_words[diagnostics_offset()].value, 1u);
    }
}

uint probe_key_low_from_meta(uint cascade, uint frame, uint probe) {
    vec4 meta = load_state_vec4(meta_base(cascade, frame, probe));
    uint flags = floatBitsToUint(meta.w);
    uint lod = flags & 7u;
    return pack_key_low(
        probe_cell(meta.xyz, cascade, lod, environment_base_spacing.w),
        lod
    );
}

uint probe_key_high_from_meta(uint cascade, uint frame, uint probe) {
    vec4 meta = load_state_vec4(meta_base(cascade, frame, probe));
    uint flags = floatBitsToUint(meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    return pack_key_high(
        probe_cell(meta.xyz, cascade, lod, environment_base_spacing.w),
        lod,
        secondary
    );
}

uint parent_key_low_from_meta(uint cascade, uint frame, uint probe) {
    vec4 meta = load_state_vec4(meta_base(cascade, frame, probe));
    uint lod = floatBitsToUint(meta.w) & 7u;
    return pack_key_low(
        probe_cell(meta.xyz, cascade + 1u, lod, environment_base_spacing.w),
        lod
    );
}

uint parent_key_high_from_meta(uint cascade, uint frame, uint probe) {
    vec4 meta = load_state_vec4(meta_base(cascade, frame, probe));
    uint flags = floatBitsToUint(meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    return pack_key_high(
        probe_cell(meta.xyz, cascade + 1u, lod, environment_base_spacing.w),
        lod,
        secondary
    );
}
@end

@block radiance_helpers
uint accum_base_frame(uint frame, uint data) {
    return radiance_layout_0.x + frame * TOTAL_VALUES * 5u + data * 5u;
}
uint accum_base(uint data) { return accum_base_frame(current_frame(), data); }
uint cone_base(uint frame, uint data) {
    return radiance_layout_0.y + frame * radiance_layout_0.z + data * 4u;
}
uint irradiance_base(uint frame, uint probe, uint texel) {
    return radiance_layout_0.w + frame * radiance_layout_1.x + (probe * 64u + texel) * 4u;
}
vec4 load_radiance_vec4(uint base) {
    return vec4(
        uintBitsToFloat(radiance_words[base].value),
        uintBitsToFloat(radiance_words[base + 1u].value),
        uintBitsToFloat(radiance_words[base + 2u].value),
        uintBitsToFloat(radiance_words[base + 3u].value)
    );
}
void store_radiance_vec4(uint base, vec4 value) {
    radiance_words[base].value = floatBitsToUint(value.x);
    radiance_words[base + 1u].value = floatBitsToUint(value.y);
    radiance_words[base + 2u].value = floatBitsToUint(value.z);
    radiance_words[base + 3u].value = floatBitsToUint(value.w);
}
@end

@block bvh_trace
vec3 decode_packed_normal(uint packed) {
    vec2 encoded = vec2(float(packed & 65535u), float(packed >> 16u)) / 65535.0;
    return oct_decode(encoded);
}

vec4 sample_material(vec2 uv, float material_index) {
    if (material_index < 0.0) return vec4(1.0);
    uint index = uint(material_index + 0.5);
    uvec2 cell = uvec2(index % 5u, index / 5u);
    vec2 atlas_uv = (vec2(cell) * 819.0 + vec2(4.0) + fract(uv) * 811.0) / 4096.0;
    return textureLod(sampler2D(material_atlas, material_sampler), atlas_uv, 0.0);
}

float aabb_near(vec3 origin, vec3 direction, bvh_node node, float maximum) {
    float near_t = 0.001;
    float far_t = maximum;
    for (uint axis = 0u; axis < 3u; axis++) {
        if (abs(direction[axis]) < 1.0e-12) {
            if (origin[axis] < node.bounds_min[axis] || origin[axis] > node.bounds_max[axis]) {
                return 3.402823466e+38;
            }
            continue;
        }
        float inverse = 1.0 / direction[axis];
        float t0 = (node.bounds_min[axis] - origin[axis]) * inverse;
        float t1 = (node.bounds_max[axis] - origin[axis]) * inverse;
        near_t = max(near_t, min(t0, t1));
        far_t = min(far_t, max(t0, t1));
        if (near_t > far_t) return 3.402823466e+38;
    }
    return near_t;
}

bool intersect_triangle(vec3 origin, vec3 direction, triangle tri, float maximum, out float distance, out vec3 barycentric) {
    vec3 p = cross(direction, tri.edge_ac.xyz);
    float determinant = dot(tri.edge_ab.xyz, p);
    if (abs(determinant) < 1.0e-8) return false;
    float inverse = 1.0 / determinant;
    vec3 t = origin - tri.a.xyz;
    float u = dot(t, p) * inverse;
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(t, tri.edge_ab.xyz);
    float v = dot(direction, q) * inverse;
    if (v < 0.0 || u + v > 1.0) return false;
    distance = dot(tri.edge_ac.xyz, q) * inverse;
    barycentric = vec3(1.0 - u - v, u, v);
    return distance >= 0.001 && distance < maximum;
}

bool trace_bvh(vec3 origin, vec3 direction, float maximum, out ray_hit result) {
    uint stack[BVH_STACK];
    uint stack_size = 1u;
    stack[0] = 0u;
    float closest = maximum;
    bool found = false;
    while (stack_size > 0u) {
        uint node_id = stack[--stack_size];
        bvh_node node = bvh_nodes[node_id];
        if (aabb_near(origin, direction, node, closest) == 3.402823466e+38) continue;
        bool leaf = (node.right_or_count & LEAF_BIT) != 0u;
        if (leaf) {
            uint count = node.right_or_count & ~LEAF_BIT;
            for (uint index = 0u; index < count; index++) {
                uint triangle_id = node.left_or_first + index;
                triangle tri = bvh_triangles[triangle_id];
                float distance;
                vec3 barycentric;
                if (intersect_triangle(origin, direction, tri, closest, distance, barycentric)) {
                    vec2 uv = tri.uv_ab.xy * barycentric.x +
                              tri.uv_ab.zw * barycentric.y +
                              tri.uv_c_material.xy * barycentric.z;
                    vec4 material = sample_material(uv, tri.uv_c_material.z);
                    if (tri.uv_c_material.w > 0.0 && material.a < tri.uv_c_material.w) {
                        continue;
                    }
                    closest = distance;
                    vec3 normal = normalize(
                        decode_packed_normal(tri.normal_oct.x) * barycentric.x +
                        decode_packed_normal(tri.normal_oct.y) * barycentric.y +
                        decode_packed_normal(tri.normal_oct.z) * barycentric.z
                    );
                    if (dot(normal, direction) > 0.0) normal = -normal;
                    result.t = distance;
                    result.normal = normal;
                    result.albedo = tri.albedo.xyz * material.rgb;
                    result.emissive = tri.emissive.xyz;
                    result.triangle_id = triangle_id;
                    found = true;
                }
            }
        } else {
            uint left = node.left_or_first;
            uint right = node.right_or_count;
            float left_near = aabb_near(origin, direction, bvh_nodes[left], closest);
            float right_near = aabb_near(origin, direction, bvh_nodes[right], closest);
            uint near_node = left_near <= right_near ? left : right;
            uint far_node = left_near <= right_near ? right : left;
            float near_t = min(left_near, right_near);
            float far_t = max(left_near, right_near);
            if (far_t < 3.402823466e+38) {
                if (stack_size < BVH_STACK) stack[stack_size++] = far_node;
                else atomicAdd(state_words[diagnostics_offset() + 3u].value, 1u);
            }
            if (near_t < 3.402823466e+38) {
                if (stack_size < BVH_STACK) stack[stack_size++] = near_node;
                else atomicAdd(state_words[diagnostics_offset() + 3u].value, 1u);
            }
        }
    }
    return found;
}

// Visibility rays only need to know whether the first alpha-opaque blocker
// exists. Returning from the first accepted leaf hit avoids traversing the
// rest of the BVH and skips material RGB/normal decoding entirely.
bool trace_bvh_any(vec3 origin, vec3 direction, float maximum) {
    uint stack[BVH_STACK];
    uint stack_size = 1u;
    stack[0] = 0u;
    while (stack_size > 0u) {
        uint node_id = stack[--stack_size];
        bvh_node node = bvh_nodes[node_id];
        if (aabb_near(origin, direction, node, maximum) == 3.402823466e+38) continue;
        bool leaf = (node.right_or_count & LEAF_BIT) != 0u;
        if (leaf) {
            uint count = node.right_or_count & ~LEAF_BIT;
            for (uint index = 0u; index < count; index++) {
                triangle tri = bvh_triangles[node.left_or_first + index];
                float distance;
                vec3 barycentric;
                if (!intersect_triangle(
                    origin,
                    direction,
                    tri,
                    maximum,
                    distance,
                    barycentric
                )) {
                    continue;
                }
                if (tri.uv_c_material.w <= 0.0) return true;
                vec2 uv = tri.uv_ab.xy * barycentric.x +
                          tri.uv_ab.zw * barycentric.y +
                          tri.uv_c_material.xy * barycentric.z;
                if (sample_material(uv, tri.uv_c_material.z).a >=
                    tri.uv_c_material.w) {
                    return true;
                }
            }
        } else {
            uint left = node.left_or_first;
            uint right = node.right_or_count;
            float left_near = aabb_near(origin, direction, bvh_nodes[left], maximum);
            float right_near = aabb_near(origin, direction, bvh_nodes[right], maximum);
            uint near_node = left_near <= right_near ? left : right;
            uint far_node = left_near <= right_near ? right : left;
            float near_t = min(left_near, right_near);
            float far_t = max(left_near, right_near);
            if (far_t < 3.402823466e+38) {
                if (stack_size < BVH_STACK) stack[stack_size++] = far_node;
                else atomicAdd(state_words[diagnostics_offset() + 3u].value, 1u);
            }
            if (near_t < 3.402823466e+38) {
                if (stack_size < BVH_STACK) stack[stack_size++] = near_node;
                else atomicAdd(state_words[diagnostics_offset() + 3u].value, 1u);
            }
        }
    }
    return false;
}
@end

@cs cs_clear_state
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < pass_params.y) state_words[pass_params.x + index].value = pass_params.z;
}
@end
@program clear_state cs_clear_state

@cs cs_clear_radiance
@include_block types
@include_block uniforms
layout(binding=1) buffer radiance_buffer { word radiance_words[]; };
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < pass_params.y) radiance_words[pass_params.x + index].value = pass_params.z;
}
@end
@program clear_radiance cs_clear_radiance

@cs cs_primary
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=2) readonly buffer bvh_node_buffer { bvh_node bvh_nodes[]; };
layout(binding=3) readonly buffer triangle_buffer { triangle bvh_triangles[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
layout(binding=7) uniform texture2D material_atlas;
layout(binding=2) uniform sampler material_sampler;
@include_block state_helpers
@include_block bvh_trace
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 size = dimensions_frame.xy;
    if (any(greaterThanEqual(pixel, size))) return;
    uint index = pixel.y * size.x + pixel.x;
    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(size);
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 near_h = inverse_view_projection * vec4(ndc, 0.0, 1.0);
    vec4 far_h = inverse_view_projection * vec4(ndc, 1.0, 1.0);
    vec3 near_p = near_h.xyz / near_h.w;
    vec3 far_p = far_h.xyz / far_h.w;
    vec3 direction = normalize(far_p - near_p);
    ray_hit hit;
    surface value;
    if (trace_bvh(camera_time.xyz, direction, 1.0e6, hit)) {
        vec3 position = camera_time.xyz + direction * hit.t;
        float sun_cosine = max(0.0, dot(hit.normal, sun_direction_intensity.xyz));
        float visibility = 1.0;
        if (sun_cosine > 0.0 &&
            trace_bvh_any(
                position + hit.normal * 0.003,
                sun_direction_intensity.xyz,
                1.0e5
            )) {
            visibility = 0.0;
        }
        vec3 to_point = point_position_intensity.xyz - position;
        float point_distance2 = max(0.25, dot(to_point, to_point));
        vec3 point_direction = to_point * inversesqrt(point_distance2);
        float point_cosine = max(0.0, dot(hit.normal, point_direction));
        float point_visibility = 1.0;
        if (point_cosine > 0.0 &&
            trace_bvh_any(
                position + hit.normal * 0.003,
                point_direction,
                max(0.001, sqrt(point_distance2) - 0.006)
            )) {
            point_visibility = 0.0;
        }
        vec3 direct = hit.emissive + hit.albedo / PI * (
            sun_direction_intensity.w * sun_cosine * visibility * vec3(1.0, 0.92, 0.78) +
            point_color_exposure.xyz * point_position_intensity.w *
                point_cosine * point_visibility / point_distance2
        );
        value.position = vec4(position, 1.0);
        value.normal_depth = vec4(hit.normal, hit.t);
        value.albedo = vec4(hit.albedo, 1.0);
        value.direct_emissive = vec4(direct, max(max(hit.emissive.r, hit.emissive.g), hit.emissive.b));
    } else {
        value.position = vec4(0.0);
        value.normal_depth = vec4(0.0);
        value.albedo = vec4(0.0);
        value.direct_emissive = vec4(environment_base_spacing.xyz, 0.0);
    }
    surfaces[index] = value;
}
@end
@program primary cs_primary

@cs cs_insert_primary
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
@include_block state_helpers
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint slot = gl_GlobalInvocationID.x;
    uint primary_slots = sample_count() * 2u;
    if (slot >= primary_slots) return;
    uint base_sample = slot % sample_count();
    uint layer = slot / sample_count();
    uint stride = dimensions_frame.w;
    uint sample_width = (dimensions_frame.x + stride - 1u) / stride;
    uvec2 sample_pixel = uvec2(base_sample % sample_width, base_sample / sample_width);
    uvec2 pixel = min(dimensions_frame.xy - uvec2(1u), sample_pixel * stride + uvec2(stride / 2u));
    surface surf = surfaces[pixel.y * dimensions_frame.x + pixel.x];
    if (surf.position.w < 0.5) return;
    float distance = max(max(abs(surf.position.x - camera_time.x), abs(surf.position.y - camera_time.y)), abs(surf.position.z - camera_time.z));
    float normalized = max(1.0, distance / (environment_base_spacing.w * 16.0));
    uint fine = uint(clamp(floor(log2(normalized)), 0.0, 7.0));
    uint coarse = min(7u, fine + 1u);
    float boundary = environment_base_spacing.w * 16.0 * exp2(float(fine + 1u));
    float coarse_weight = clamp((distance - boundary * 0.9) / (boundary * 0.1), 0.0, 1.0);
    if (layer == 1u && (coarse == fine || coarse_weight <= 0.0)) return;
    uint lod = layer == 0u ? fine : coarse;
    ivec3 cell = probe_cell(surf.position.xyz, 0u, lod, environment_base_spacing.w);
    uint key_low = pack_key_low(cell, lod);
    uint key_high = pack_key_high(cell, lod, false);
    uint deferred;
    insert_key(0u, key_low, key_high, deferred);
    if (deferred != 0u) {
        enqueue_retry(0u, slot);
    }
}
@end
@program insert_primary cs_insert_primary

@cs cs_insert_secondary
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint secondary_slot = gl_GlobalInvocationID.x;
    if (secondary_slot >= sample_count() || dimensions_frame.z == 0u) return;
    uint sample_id = secondary_slot;
    uint record = hit_record_base(previous_frame(), sample_id);
    vec4 position = load_state_vec4(record);
    vec4 normal_lod = load_state_vec4(record + 4u);
    if (position.w < 0.5 || normal_lod.w < 0.5) return;
    uint lod = min(7u, uint(normal_lod.w) + 2u);
    ivec3 cell = probe_cell(position.xyz, 0u, lod, environment_base_spacing.w);
    uint key_low = pack_key_low(cell, lod);
    uint key_high = pack_key_high(cell, lod, true);
    uint deferred;
    insert_key(0u, key_low, key_high, deferred);
    if (deferred != 0u) {
        enqueue_retry(0u, sample_count() * 2u + secondary_slot);
    }
}
@end
@program insert_secondary cs_insert_secondary

@cs cs_insert_retry
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
@include_block state_helpers
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint retry_index = gl_GlobalInvocationID.x;
    uint read_queue = pass_params.x;
    uint write_queue = pass_params.y;
    uint retry_count = min(
        state_words[retry_counter(read_queue)].value,
        ray_slot_count()
    );
    if (retry_index >= retry_count) return;
    uint source = state_words[retry_base(read_queue) + retry_index].value;
    bool deferred = false;
    if (source < sample_count() * 2u) {
        uint base_sample = source % sample_count();
        uint layer = source / sample_count();
        uint sample_width = dimensions_frame.x;
        uvec2 pixel = uvec2(base_sample % sample_width, base_sample / sample_width);
        surface surf = surfaces[pixel.y * dimensions_frame.x + pixel.x];
        if (surf.position.w < 0.5) return;
        float distance = max(
            max(abs(surf.position.x - camera_time.x), abs(surf.position.y - camera_time.y)),
            abs(surf.position.z - camera_time.z)
        );
        uint fine = uint(clamp(
            floor(log2(max(1.0, distance / (environment_base_spacing.w * 16.0)))),
            0.0,
            7.0
        ));
        uint coarse = min(7u, fine + 1u);
        float boundary = environment_base_spacing.w * 16.0 * exp2(float(fine + 1u));
        float coarse_weight = clamp(
            (distance - boundary * 0.9) / (boundary * 0.1),
            0.0,
            1.0
        );
        if (layer == 1u && (coarse == fine || coarse_weight <= 0.0)) return;
        uint lod = layer == 0u ? fine : coarse;
        ivec3 cell = probe_cell(surf.position.xyz, 0u, lod, environment_base_spacing.w);
        uint retry_deferred;
        insert_key(
            0u,
            pack_key_low(cell, lod),
            pack_key_high(cell, lod, false),
            retry_deferred
        );
        deferred = retry_deferred != 0u;
    } else {
        uint sample_id = source - sample_count() * 2u;
        if (sample_id >= sample_count() || dimensions_frame.z == 0u) return;
        uint record = hit_record_base(previous_frame(), sample_id);
        vec4 position = load_state_vec4(record);
        vec4 normal_lod = load_state_vec4(record + 4u);
        if (position.w < 0.5 || normal_lod.w < 0.5) return;
        uint lod = min(7u, uint(normal_lod.w) + 2u);
        ivec3 cell = probe_cell(position.xyz, 0u, lod, environment_base_spacing.w);
        uint retry_deferred;
        insert_key(
            0u,
            pack_key_low(cell, lod),
            pack_key_high(cell, lod, true),
            retry_deferred
        );
        deferred = retry_deferred != 0u;
    }
    if (deferred) {
        if (pass_params.z != 0u) {
            atomicAdd(state_words[diagnostics_offset()].value, 1u);
        } else {
            enqueue_retry(write_queue, source);
        }
    }
}
@end
@program insert_retry cs_insert_retry

@cs cs_canonicalize
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint cascade = pass_params.x;
    uint slot = gl_GlobalInvocationID.x;
    if (cascade >= 4u || slot >= HASH_CAPS[cascade]) return;
    uint base = hash_base(cascade, current_frame());
    uint address = base + slot * 5u;
    if (state_words[address].value == EMPTY ||
        state_words[address + 4u].value != INDEX_RESERVED) return;
    uint token = state_words[address + 1u].value;
    uint key_low = state_words[address + 2u].value;
    uint key_high = state_words[address + 3u].value;
    uint mask = HASH_CAPS[cascade] - 1u;
    uint start = token & mask;
    // A thread that encountered an unpublished reservation during collection may have
    // published the same exact key into a later slot. Canonicalize only the
    // earliest full-key match, while diagnosing true token collisions.
    for (uint attempt = 0u; attempt < 32u; attempt++) {
        uint earlier_slot = (start + attempt) & mask;
        if (earlier_slot == slot) break;
        uint earlier = base + earlier_slot * 5u;
        if (state_words[earlier].value == EMPTY ||
            state_words[earlier + 4u].value == EMPTY ||
            state_words[earlier + 1u].value != token) {
            continue;
        }
        uint earlier_low = state_words[earlier + 2u].value;
        uint earlier_high = state_words[earlier + 3u].value;
        if (earlier_low == key_low && earlier_high == key_high) {
            atomicExchange(
                state_words[address + 4u].value,
                INDEX_DUPLICATE
            );
            memoryBarrierBuffer();
            // Standard open-addressing tombstone: the duplicate no longer
            // participates as an occupied key, but lookup must continue past
            // it to keys later in the probe chain.
            atomicExchange(
                state_words[address].value,
                SLOT_TOMBSTONE
            );
            return;
        }
        atomicAdd(state_words[diagnostics_offset() + 1u].value, 1u);
    }
    uint probe = atomicAdd(state_words[counter_active(cascade)].value, 1u);
    if (probe >= PROBE_CAPS[cascade]) {
        atomicAdd(state_words[diagnostics_offset()].value, 1u);
        atomicExchange(state_words[address + 4u].value, EMPTY);
        return;
    }
    uint x = key_low & 0x7ffffu;
    uint y = ((key_low >> 19u) & 0x1fffu) | ((key_high & 0x3fu) << 13u);
    uint z = (key_high >> 6u) & 0x7ffffu;
    ivec3 cell = ivec3(x, y, z) - ivec3(262144);
    uint lod = (key_high >> 25u) & 7u;
    bool secondary = (key_high & (1u << 28u)) != 0u;
    uint flags = lod | (secondary ? (1u << 30u) : 0u);
    uint meta = meta_base(cascade, current_frame(), probe);
    store_state_vec4(
        meta,
        vec4(
            probe_position(cell, cascade, lod, environment_base_spacing.w),
            uintBitsToFloat(flags)
        )
    );
    atomicExchange(state_words[address + 4u].value, probe);
}
@end
@program canonicalize cs_canonicalize

@cs cs_map_primary
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
@include_block state_helpers
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= sample_count() * 2u) return;
    uint sample_id = slot % sample_count();
    uint layer = slot / sample_count();
    uint sample_width = dimensions_frame.x;
    uvec2 pixel = uvec2(sample_id % sample_width, sample_id / sample_width);
    surface surf = surfaces[pixel.y * dimensions_frame.x + pixel.x];
    if (surf.position.w < 0.5) return;
    float distance = max(
        max(abs(surf.position.x - camera_time.x), abs(surf.position.y - camera_time.y)),
        abs(surf.position.z - camera_time.z)
    );
    uint fine = uint(clamp(
        floor(log2(max(1.0, distance / (environment_base_spacing.w * 16.0)))),
        0.0,
        7.0
    ));
    uint coarse = min(7u, fine + 1u);
    float boundary = environment_base_spacing.w * 16.0 * exp2(float(fine + 1u));
    float coarse_weight = clamp((distance - boundary * 0.9) / (boundary * 0.1), 0.0, 1.0);
    if (layer == 1u && (coarse == fine || coarse_weight <= 0.0)) return;
    uint lod = layer == 0u ? fine : coarse;
    ivec3 cell = probe_cell(surf.position.xyz, 0u, lod, environment_base_spacing.w);
    uint probe = lookup_probe(
        0u,
        pack_key_low(cell, lod),
        pack_key_high(cell, lod, false)
    );
    if (probe == EMPTY) return;
    state_words[ray_map_base(slot)].value = probe;
    state_words[ray_map_base(slot) + 1u].value = lod;
    atomicAdd(state_words[block_base(probe, slot / RANK_BLOCK_SIZE)].value, 1u);
}
@end
@program map_primary cs_map_primary

@cs cs_map_secondary
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
void main() {
    uint sample_id = gl_GlobalInvocationID.x;
    if (sample_id >= sample_count() || dimensions_frame.z == 0u) return;
    uint record = hit_record_base(previous_frame(), sample_id);
    vec4 position = load_state_vec4(record);
    vec4 normal_lod = load_state_vec4(record + 4u);
    if (position.w < 0.5 || normal_lod.w < 0.5) return;
    uint lod = min(7u, uint(normal_lod.w) + 2u);
    ivec3 cell = probe_cell(position.xyz, 0u, lod, environment_base_spacing.w);
    uint probe = lookup_probe(
        0u,
        pack_key_low(cell, lod),
        pack_key_high(cell, lod, true)
    );
    if (probe == EMPTY) return;
    uint slot = sample_count() * 2u + sample_id;
    uint flags = lod | (1u << 30u);
    state_words[ray_map_base(slot)].value = probe;
    state_words[ray_map_base(slot) + 1u].value = flags;
    atomicAdd(state_words[block_base(probe, slot / RANK_BLOCK_SIZE)].value, 1u);
}
@end
@program map_secondary cs_map_secondary

@cs cs_prefix_blocks
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint probe = gl_GlobalInvocationID.x;
    uint active_count = min(state_words[counter_active(0u)].value, PROBE_CAPS[0]);
    if (probe >= active_count) return;
    uint prefix = 0u;
    for (uint block = 0u; block < state_layout_1.w; block++) {
        uint address = block_base(probe, block);
        uint count = state_words[address].value;
        state_words[address].value = prefix;
        prefix += count;
    }
    state_words[counter_ray_count(0u, probe)].value = prefix;
}
@end
@program prefix_blocks cs_prefix_blocks

@cs cs_insert_parent
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint child = gl_GlobalInvocationID.x;
    uint cascade = pass_params.x;
    if (cascade == 0u || cascade >= 4u) return;
    uint child_cascade = cascade - 1u;
    uint active_count = min(state_words[counter_active(child_cascade)].value, PROBE_CAPS[child_cascade]);
    if (child >= active_count) return;
    vec4 child_meta = load_state_vec4(meta_base(child_cascade, current_frame(), child));
    uint flags = floatBitsToUint(child_meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    ivec3 cell = probe_cell(child_meta.xyz, cascade, lod, environment_base_spacing.w);
    uint key_low = pack_key_low(cell, lod);
    uint key_high = pack_key_high(cell, lod, secondary);
    uint deferred;
    insert_key(cascade, key_low, key_high, deferred);
    if (deferred != 0u) {
        enqueue_retry(0u, child);
    }
}
@end
@program insert_parent cs_insert_parent

@cs cs_insert_parent_retry
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint retry_index = gl_GlobalInvocationID.x;
    uint cascade = pass_params.x;
    uint read_queue = pass_params.y;
    uint write_queue = pass_params.z;
    uint retry_count = min(
        state_words[retry_counter(read_queue)].value,
        ray_slot_count()
    );
    if (retry_index >= retry_count || cascade == 0u || cascade >= 4u) return;
    uint child = state_words[retry_base(read_queue) + retry_index].value;
    uint child_cascade = cascade - 1u;
    uint active_count = min(
        state_words[counter_active(child_cascade)].value,
        PROBE_CAPS[child_cascade]
    );
    if (child >= active_count) return;
    vec4 child_meta = load_state_vec4(meta_base(child_cascade, current_frame(), child));
    uint flags = floatBitsToUint(child_meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    ivec3 cell = probe_cell(child_meta.xyz, cascade, lod, environment_base_spacing.w);
    uint retry_deferred;
    insert_key(
        cascade,
        pack_key_low(cell, lod),
        pack_key_high(cell, lod, secondary),
        retry_deferred
    );
    bool deferred = retry_deferred != 0u;
    if (deferred) {
        if (pass_params.w != 0u) {
            atomicAdd(state_words[diagnostics_offset()].value, 1u);
        } else {
            enqueue_retry(write_queue, child);
        }
    }
}
@end
@program insert_parent_retry cs_insert_parent_retry

@cs cs_propagate_parent
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint child = gl_GlobalInvocationID.x;
    uint cascade = pass_params.x;
    if (cascade == 0u || cascade >= 4u) return;
    uint child_cascade = cascade - 1u;
    uint active_count = min(
        state_words[counter_active(child_cascade)].value,
        PROBE_CAPS[child_cascade]
    );
    if (child >= active_count) return;
    vec4 child_meta = load_state_vec4(meta_base(child_cascade, current_frame(), child));
    uint flags = floatBitsToUint(child_meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    ivec3 cell = probe_cell(child_meta.xyz, cascade, lod, environment_base_spacing.w);
    uint parent = lookup_probe(
        cascade,
        pack_key_low(cell, lod),
        pack_key_high(cell, lod, secondary)
    );
    if (parent != EMPTY) {
        atomicAdd(
            state_words[counter_ray_count(cascade, parent)].value,
            state_words[counter_ray_count(child_cascade, child)].value
        );
    }
}
@end
@program propagate_parent cs_propagate_parent

@cs cs_assign_offsets
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
@include_block state_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint probe = gl_GlobalInvocationID.x;
    uint cascade = pass_params.x;
    uint active_count = min(state_words[counter_active(cascade)].value, PROBE_CAPS[cascade]);
    if (probe >= active_count) return;
    uint key_low = probe_key_low_from_meta(cascade, current_frame(), probe);
    uint key_high = probe_key_high_from_meta(cascade, current_frame(), probe);
    uint offset = 0u;
    if (cascade == 3u) {
        // The top level is capped at 256 probes, so a key-canonical scan is
        // bounded to 65k comparisons for the whole dispatch.
        for (uint other = 0u; other < active_count; other++) {
            uint other_low = probe_key_low_from_meta(cascade, current_frame(), other);
            uint other_high = probe_key_high_from_meta(cascade, current_frame(), other);
            if (key_less(other_low, other_high, key_low, key_high)) {
                offset += state_words[counter_ray_count(cascade, other)].value;
            }
        }
    } else {
        uint parent_low = parent_key_low_from_meta(cascade, current_frame(), probe);
        uint parent_high = parent_key_high_from_meta(cascade, current_frame(), probe);
        uint parent = lookup_probe(cascade + 1u, parent_low, parent_high);
        if (parent != EMPTY) offset = state_words[counter_ray_offset(cascade + 1u, parent)].value;
        vec4 meta = load_state_vec4(meta_base(cascade, current_frame(), probe));
        uint flags = floatBitsToUint(meta.w);
        uint lod = flags & 7u;
        bool secondary = (flags & (1u << 30u)) != 0u;
        ivec3 parent_cell = probe_cell(
            meta.xyz,
            cascade + 1u,
            lod,
            environment_base_spacing.w
        );
        // Adjacent cascade spacing differs by exactly two, so a parent has
        // only eight possible sparse children. Looking them up directly makes
        // this pass O(8N), while preserving the exact key ordering required by
        // Algorithm 3.
        for (uint child = 0u; child < 8u; child++) {
            ivec3 bits = ivec3(
                int(child & 1u),
                int((child >> 1u) & 1u),
                int((child >> 2u) & 1u)
            );
            ivec3 child_cell = parent_cell * 2 + bits;
            uint child_low = pack_key_low(child_cell, lod);
            uint child_high = pack_key_high(child_cell, lod, secondary);
            if (!key_less(child_low, child_high, key_low, key_high)) continue;
            uint sibling = lookup_probe(cascade, child_low, child_high);
            if (sibling != EMPTY && sibling < PROBE_CAPS[cascade]) {
                offset += state_words[counter_ray_count(cascade, sibling)].value;
            }
        }
    }
    state_words[counter_ray_offset(cascade, probe)].value = offset;
}
@end
@program assign_offsets cs_assign_offsets

@cs cs_trace_split
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=1) buffer radiance_buffer { word radiance_words[]; };
layout(binding=2) readonly buffer bvh_node_buffer { bvh_node bvh_nodes[]; };
layout(binding=3) readonly buffer triangle_buffer { triangle bvh_triangles[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
layout(binding=7) uniform texture2D material_atlas;
layout(binding=2) uniform sampler material_sampler;
@include_block state_helpers
@include_block radiance_helpers
@include_block bvh_trace

vec3 previous_irradiance(vec3 position, vec3 normal, uint lod) {
    ivec3 cell = probe_cell(position, 0u, lod, environment_base_spacing.w);
    uint key_low = pack_key_low(cell, lod);
    uint key_high = pack_key_high(cell, lod, true);
    uint probe = lookup_probe_frame(0u, key_low, key_high, previous_frame());
    if (probe == EMPTY || probe >= PROBE_CAPS[0]) return vec3(0.0);
    vec2 oct = oct_encode(normal);
    uvec2 texel = uvec2(clamp(floor(oct * 6.0), vec2(0.0), vec2(5.0))) + uvec2(1u);
    return load_radiance_vec4(irradiance_base(previous_frame(), probe, texel.y * 8u + texel.x)).xyz;
}

vec3 hit_radiance(vec3 position, ray_hit hit, uint lod) {
    float sun_cosine = max(0.0, dot(hit.normal, sun_direction_intensity.xyz));
    float sun_visibility = 1.0;
    if (sun_cosine > 0.0 &&
        trace_bvh_any(
            position + hit.normal * 0.003,
            sun_direction_intensity.xyz,
            1.0e5
        )) {
        sun_visibility = 0.0;
    }
    vec3 to_point = point_position_intensity.xyz - position;
    float point_distance2 = max(0.25, dot(to_point, to_point));
    vec3 point_direction = to_point * inversesqrt(point_distance2);
    float point_cosine = max(0.0, dot(hit.normal, point_direction));
    float point_visibility = 1.0;
    if (point_cosine > 0.0 &&
        trace_bvh_any(
            position + hit.normal * 0.003,
            point_direction,
            max(0.001, sqrt(point_distance2) - 0.006)
        )) {
        point_visibility = 0.0;
    }
    vec3 direct = hit.emissive + hit.albedo / PI *
        (
            sun_direction_intensity.w * sun_cosine * sun_visibility *
                vec3(1.0, 0.92, 0.78) +
            point_color_exposure.xyz * point_position_intensity.w *
                point_cosine * point_visibility / point_distance2
        );
    return direct + hit.albedo / PI * previous_irradiance(position, hit.normal, min(7u, lod + 2u));
}

void deposit(uint cascade, uint probe, uint direction, vec3 radiance, float beta) {
    uint base = accum_base(data_index(cascade, probe, direction));
    atomicAdd(radiance_words[base].value, fixed_value(radiance.x));
    atomicAdd(radiance_words[base + 1u].value, fixed_value(radiance.y));
    atomicAdd(radiance_words[base + 2u].value, fixed_value(radiance.z));
    atomicAdd(radiance_words[base + 3u].value, fixed_value(beta));
    atomicAdd(radiance_words[base + 4u].value, 1u);
}

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= ray_slot_count()) return;
    uint probe0 = state_words[ray_map_base(slot)].value;
    if (probe0 == EMPTY || probe0 >= PROBE_CAPS[0]) return;
    uint flags = state_words[ray_map_base(slot) + 1u].value;
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    uint base_sample = slot % sample_count();
    vec3 origin;
    vec3 normal;
    if (!secondary) {
        uint stride = dimensions_frame.w;
        uint sample_width = (dimensions_frame.x + stride - 1u) / stride;
        uvec2 sample_pixel = uvec2(base_sample % sample_width, base_sample / sample_width);
        uvec2 pixel = min(dimensions_frame.xy - uvec2(1u), sample_pixel * stride + uvec2(stride / 2u));
        surface surf = surfaces[pixel.y * dimensions_frame.x + pixel.x];
        if (surf.position.w < 0.5) return;
        origin = surf.position.xyz;
        normal = normalize(surf.normal_depth.xyz);
    } else {
        uint record = hit_record_base(previous_frame(), base_sample);
        vec4 position = load_state_vec4(record);
        vec4 normal_lod = load_state_vec4(record + 4u);
        if (position.w < 0.5) return;
        origin = position.xyz;
        normal = normalize(normal_lod.xyz);
    }
    uint block = slot / RANK_BLOCK_SIZE;
    uint local_rank = state_words[block_base(probe0, block)].value;
    uint first = block * RANK_BLOCK_SIZE;
    for (uint other = first; other < slot; other++) {
        if (state_words[ray_map_base(other)].value == probe0) local_rank++;
    }
    uint sequence = state_words[counter_ray_offset(0u, probe0)].value + local_rank;
    uint rotation_seed;
    if (radiance_layout_1.y != 0u) {
        rotation_seed = dimensions_frame.z;
    } else {
        uint seed_low = probe_key_low_from_meta(
            0u,
            current_frame(),
            probe0
        );
        uint seed_high = probe_key_high_from_meta(
            0u,
            current_frame(),
            probe0
        );
        rotation_seed = key_hash(seed_low, seed_high);
    }
    vec2 jitter = vec2(
        float(hash32(rotation_seed ^ 0x9e3779b9u) & 0xffffu),
        float(hash32(rotation_seed ^ 0x243f6a88u) & 0xffffu)
    ) / 65536.0;
    vec3 direction = decode_equal_area(r2(sequence, jitter));
    if (dot(direction, normal) < 0.0) direction = -direction;
    ray_hit hit;
    float near_offset = pass_params.z != 0u
        ? environment_base_spacing.w * exp2(float(lod))
        : 0.0;
    vec3 trace_origin = origin + normal * 0.003 + direction * near_offset;
    bool has_hit = trace_bvh(trace_origin, direction, 1.0e6, hit);
    float distance = has_hit ? hit.t + 0.003 : 1.0e20;
    vec3 radiance = has_hit ? hit_radiance(trace_origin + direction * hit.t, hit, lod)
                            : environment_base_spacing.xyz;
    uint target = 0u;
    distance += near_offset;
    while (target < 3u && distance > interval_cutoff(environment_base_spacing.w, target, lod)) target++;
    for (uint cascade = 0u; cascade < 4u; cascade++) {
        ivec3 cell = probe_cell(origin, cascade, lod, environment_base_spacing.w);
        uint key_low = pack_key_low(cell, lod);
        uint key_high = pack_key_high(cell, lod, secondary);
        uint probe = lookup_probe(cascade, key_low, key_high);
        if (probe == EMPTY || probe >= PROBE_CAPS[cascade]) continue;
        uint direction_id = direction_index(direction, cascade);
        if (cascade < target) deposit(cascade, probe, direction_id, vec3(0.0), 1.0);
        else if (cascade == target) deposit(cascade, probe, direction_id, radiance, 0.0);
    }
    uint primary_layer = slot / sample_count();
    if (!secondary && primary_layer == 0u) {
        uint record = hit_record_base(current_frame(), base_sample);
        if (has_hit) {
            vec3 hit_position = trace_origin + direction * hit.t;
            store_state_vec4(record, vec4(hit_position, 1.0));
            store_state_vec4(record + 4u, vec4(hit.normal, float(lod)));
        } else {
            store_state_vec4(record, vec4(0.0));
            store_state_vec4(record + 4u, vec4(0.0));
        }
    }
}
@end
@program trace_split cs_trace_split

@cs cs_merge
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=1) buffer radiance_buffer { word radiance_words[]; };
@include_block state_helpers
@include_block radiance_helpers

vec4 sample_parent_direction(uint cascade, vec3 position, uint lod, bool secondary, uint direction) {
    uint parent = cascade + 1u;
    float spacing = environment_base_spacing.w * exp2(float(parent + lod));
    vec3 grid = position / spacing - vec3(0.5);
    ivec3 cell = ivec3(floor(grid));
    vec3 fraction = fract(grid);
    vec3 value = vec3(0.0);
    float total = 0.0;
    for (uint corner = 0u; corner < 8u; corner++) {
        ivec3 bits = ivec3(int(corner & 1u), int((corner >> 1u) & 1u), int((corner >> 2u) & 1u));
        vec3 weights = mix(vec3(1.0) - fraction, fraction, vec3(bits));
        float weight = weights.x * weights.y * weights.z;
        ivec3 corner_cell = cell + bits;
        uint key_low = pack_key_low(corner_cell, lod);
        uint key_high = pack_key_high(corner_cell, lod, secondary);
        uint probe = lookup_probe(parent, key_low, key_high);
        if (probe != EMPTY && probe < PROBE_CAPS[parent]) {
            uint parent_data = data_index(parent, probe, direction);
            if (radiance_words[accum_base(parent_data) + 4u].value == 0u) {
                continue;
            }
            value += load_radiance_vec4(
                cone_base(current_frame(), parent_data)
            ).xyz * weight;
            total += weight;
        }
    }
    return total > 1.0e-5 ? vec4(value / total, 1.0) : vec4(0.0);
}

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint cascade = pass_params.x;
    uint linear = gl_GlobalInvocationID.x;
    uint direction = linear % DIR_COUNTS[cascade];
    uint probe = linear / DIR_COUNTS[cascade];
    uint active_count = min(state_words[counter_active(cascade)].value, PROBE_CAPS[cascade]);
    if (probe >= active_count) return;
    uint data = data_index(cascade, probe, direction);
    uint accum = accum_base(data);
    uint count = radiance_words[accum + 4u].value;
    uint resolved_count = count;
    vec3 interval = vec3(0.0);
    float beta = 1.0;
    bool valid = count > 0u;
    if (count > 0u) {
        float denominator = FIXED_SCALE * float(count);
        interval = vec3(radiance_words[accum].value, radiance_words[accum + 1u].value, radiance_words[accum + 2u].value) / denominator;
        beta = float(radiance_words[accum + 3u].value) / denominator;
    }
    vec4 meta = load_state_vec4(meta_base(cascade, current_frame(), probe));
    uint flags = floatBitsToUint(meta.w);
    uint lod = flags & 7u;
    bool secondary = (flags & (1u << 30u)) != 0u;
    uint key_low = probe_key_low_from_meta(cascade, current_frame(), probe);
    uint key_high = probe_key_high_from_meta(cascade, current_frame(), probe);
    uint old_probe = lookup_probe_frame(cascade, key_low, key_high, previous_frame());
    if (dimensions_frame.z > 0u && old_probe != EMPTY && old_probe < PROBE_CAPS[cascade]) {
        uint previous_accum = accum_base_frame(
            previous_frame(),
            data_index(cascade, old_probe, direction)
        );
        uint previous_count = radiance_words[previous_accum + 4u].value;
        if (previous_count > 0u) {
            valid = true;
            float previous_denominator = FIXED_SCALE * float(previous_count);
            vec3 previous_interval = vec3(
                radiance_words[previous_accum].value,
                radiance_words[previous_accum + 1u].value,
                radiance_words[previous_accum + 2u].value
            ) / previous_denominator;
            float previous_beta =
                float(radiance_words[previous_accum + 3u].value) / previous_denominator;
            if (pass_params.y != 0u) {
                float weight = clamp(float(pass_params.w) / 1000.0, 0.0, 0.99);
                if (count > 0u) {
                    interval = mix(interval, previous_interval, weight);
                    beta = mix(beta, previous_beta, weight);
                } else {
                    interval = previous_interval;
                    beta = previous_beta;
                }
                resolved_count = 1u;
            } else {
                uint history_samples = min(previous_count, 16384u);
                if (count > 0u) {
                    float current_samples = float(count);
                    float total_samples =
                        current_samples + float(history_samples);
                    interval = (
                        interval * current_samples +
                        previous_interval * float(history_samples)
                    ) / total_samples;
                    beta = (
                        beta * current_samples +
                        previous_beta * float(history_samples)
                    ) / total_samples;
                    resolved_count = min(16384u, count + history_samples);
                } else {
                    interval = previous_interval;
                    beta = previous_beta;
                    resolved_count = history_samples;
                }
            }
        }
    }
    if (!valid) {
        // A zero-count direction is missing data, not a transparent ray.
        // Keeping it invalid prevents parent/environment energy fabrication.
        store_radiance_vec4(
            cone_base(current_frame(), data),
            vec4(0.0)
        );
        return;
    }
    // Persist the temporally filtered directional interval itself. Reusing the
    // already-composed cone would feed parent radiance back into the near
    // interval and is a source of view-dependent energy flicker.
    resolved_count = max(1u, resolved_count);
    float stored_scale = FIXED_SCALE * float(resolved_count);
    radiance_words[accum].value = uint(round(
        clamp(interval.x, 0.0, MAX_RADIANCE) * stored_scale
    ));
    radiance_words[accum + 1u].value = uint(round(
        clamp(interval.y, 0.0, MAX_RADIANCE) * stored_scale
    ));
    radiance_words[accum + 2u].value = uint(round(
        clamp(interval.z, 0.0, MAX_RADIANCE) * stored_scale
    ));
    radiance_words[accum + 3u].value = uint(round(
        clamp(beta, 0.0, 1.0) * stored_scale
    ));
    radiance_words[accum + 4u].value = resolved_count;
    vec3 distant = environment_base_spacing.xyz;
    if (cascade < 3u) {
        distant = vec3(0.0);
        float distant_count = 0.0;
        for (uint child = 0u; child < 4u; child++) {
            vec4 sample_value = sample_parent_direction(
                cascade,
                meta.xyz,
                lod,
                secondary,
                direction * 4u + child
            );
            distant += sample_value.xyz;
            distant_count += sample_value.w;
        }
        if (distant_count > 0.0) distant /= distant_count;
    }
    store_radiance_vec4(
        cone_base(current_frame(), data),
        vec4(clamp(interval + clamp(beta, 0.0, 1.0) * distant, vec3(0.0), vec3(MAX_RADIANCE)), clamp(beta, 0.0, 1.0))
    );
}
@end
@program merge cs_merge

@cs cs_irradiance
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=1) buffer radiance_buffer { word radiance_words[]; };
layout(rgba16f, binding=6) uniform writeonly image2D irradiance_atlas_storage;
@include_block state_helpers
@include_block radiance_helpers
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
void main() {
    uint linear = gl_GlobalInvocationID.x;
    uint texel = linear & 63u;
    uint probe = linear >> 6u;
    uint active_count = min(state_words[counter_active(0u)].value, PROBE_CAPS[0]);
    if (probe >= active_count) return;
    uvec2 coord = uvec2(texel & 7u, texel >> 3u);
    // Evaluate the octahedral extension in the one-texel border. This is the
    // seam-copying construction required for filtered octahedral lookup.
    vec3 normal = oct_decode((vec2(coord) - vec2(0.5)) / 6.0);
    vec3 irradiance = vec3(0.0);
    float weight_sum = 0.0;
    for (uint direction = 0u; direction < 32u; direction++) {
        vec3 ray = direction_center(direction, 0u);
        float weight = max(0.0, dot(normal, ray));
        uint direction_data = data_index(0u, probe, direction);
        if (radiance_words[accum_base(direction_data) + 4u].value == 0u) {
            continue;
        }
        irradiance += load_radiance_vec4(
            cone_base(current_frame(), direction_data)
        ).xyz * weight;
        weight_sum += weight;
    }
    irradiance *= PI / max(1.0e-5, weight_sum);
    vec4 stored = vec4(irradiance, 1.0);
    store_radiance_vec4(irradiance_base(current_frame(), probe, texel), stored);
    uvec2 tile = uvec2(probe & 63u, probe >> 6u) * 8u;
    ivec2 atlas_coord = ivec2(tile + coord + uvec2(0u, current_frame() * 512u));
    imageStore(irradiance_atlas_storage, atlas_coord, stored);
}
@end
@program irradiance cs_irradiance

@cs cs_shade
@include_block types
@include_block uniforms
layout(binding=0) buffer state_buffer { word state_words[]; };
layout(binding=1) buffer radiance_buffer { word radiance_words[]; };
layout(binding=4) buffer gbuffer_buffer { surface surfaces[]; };
layout(rgba8, binding=5) uniform writeonly image2D output_image;
layout(binding=8) uniform texture2D irradiance_atlas;
layout(binding=3) uniform sampler irradiance_sampler;
@include_block state_helpers
@include_block radiance_helpers

vec3 sample_irradiance_lod(vec3 position, vec3 normal, uint lod) {
    float spacing = environment_base_spacing.w * exp2(float(lod));
    vec3 grid = position / spacing - vec3(0.5);
    ivec3 cell = ivec3(floor(grid));
    vec3 fraction = fract(grid);
    vec2 oct = oct_encode(normal);
    vec2 atlas_coordinate = clamp(oct * 6.0 + vec2(0.5), vec2(0.0), vec2(7.0));
    vec3 value = vec3(0.0);
    float total = 0.0;
    for (uint corner = 0u; corner < 8u; corner++) {
        ivec3 bits = ivec3(int(corner & 1u), int((corner >> 1u) & 1u), int((corner >> 2u) & 1u));
        vec3 weights = mix(vec3(1.0) - fraction, fraction, vec3(bits));
        float weight = weights.x * weights.y * weights.z;
        ivec3 corner_cell = cell + bits;
        uint key_low = pack_key_low(corner_cell, lod);
        uint key_high = pack_key_high(corner_cell, lod, false);
        uint probe = lookup_probe(0u, key_low, key_high);
        if (probe != EMPTY && probe < PROBE_CAPS[0]) {
            vec2 tile = vec2(float(probe & 63u) * 8.0, float(probe >> 6u) * 8.0);
            tile.y += float(current_frame() * 512u);
            vec2 atlas_uv = (tile + atlas_coordinate + vec2(0.5)) / vec2(512.0, 1024.0);
            vec3 filtered = textureLod(
                sampler2D(irradiance_atlas, irradiance_sampler),
                atlas_uv,
                0.0
            ).xyz;
            value += filtered * weight;
            total += weight;
        }
    }
    return total > 1.0e-5 ? value / total : vec3(0.0);
}

vec4 sample_directional_lod(
    vec3 position,
    vec3 direction,
    uint cascade,
    uint lod,
    bool composed
) {
    float spacing = environment_base_spacing.w * exp2(float(cascade + lod));
    vec3 grid = position / spacing - vec3(0.5);
    ivec3 cell = ivec3(floor(grid));
    vec3 fraction = fract(grid);
    uint direction_id = direction_index(direction, cascade);
    vec4 value = vec4(0.0);
    float total = 0.0;
    for (uint corner = 0u; corner < 8u; corner++) {
        ivec3 bits = ivec3(int(corner & 1u), int((corner >> 1u) & 1u), int((corner >> 2u) & 1u));
        vec3 weights = mix(vec3(1.0) - fraction, fraction, vec3(bits));
        float weight = weights.x * weights.y * weights.z;
        ivec3 corner_cell = cell + bits;
        uint key_low = pack_key_low(corner_cell, lod);
        uint key_high = pack_key_high(corner_cell, lod, false);
        uint probe = lookup_probe(cascade, key_low, key_high);
        if (probe == EMPTY || probe >= PROBE_CAPS[cascade]) continue;
        uint data = data_index(cascade, probe, direction_id);
        uint count = radiance_words[accum_base(data) + 4u].value;
        if (count == 0u) continue;
        if (composed) {
            value += load_radiance_vec4(cone_base(current_frame(), data)) * weight;
        } else {
            uint base = accum_base(data);
            float denominator = FIXED_SCALE * float(count);
            value += vec4(
                vec3(
                    radiance_words[base].value,
                    radiance_words[base + 1u].value,
                    radiance_words[base + 2u].value
                ) / denominator,
                float(radiance_words[base + 3u].value) / denominator
            ) * weight;
        }
        total += weight;
    }
    if (total > 1.0e-5) return value / total;
    return vec4(0.0);
}

vec3 sample_rough_specular(vec3 position, vec3 normal, vec3 view, uint lod) {
    vec3 reflected = normalize(reflect(-view, normal));
    // Section 7's rough-specular extension starts from a coarser directional
    // cone, then composes the two near raw intervals front-to-back.
    vec3 radiance = sample_directional_lod(position, reflected, 2u, lod, true).xyz;
    for (int cascade = 1; cascade >= 0; cascade--) {
        vec4 interval = sample_directional_lod(
            position,
            reflected,
            uint(cascade),
            lod,
            false
        );
        radiance = interval.xyz + clamp(interval.w, 0.0, 1.0) * radiance;
    }
    float cosine = max(0.0, dot(normal, view));
    vec3 fresnel = vec3(0.04) + vec3(0.96) * pow(1.0 - cosine, 5.0);
    return radiance * fresnel * 0.45;
}

vec4 screen_space_near_interval(vec3 position, vec3 direction, uint lod) {
    float near_length = environment_base_spacing.w * exp2(float(lod));
    for (uint step = 1u; step <= 8u; step++) {
        float distance = near_length * float(step) / 8.0;
        vec3 expected = position + direction * distance;
        vec4 clip = view_projection * vec4(expected, 1.0);
        if (clip.w <= 0.0) continue;
        vec2 uv = clip.xy / clip.w * 0.5 + vec2(0.5);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
        uvec2 pixel = min(
            dimensions_frame.xy - uvec2(1u),
            uvec2(uv * vec2(dimensions_frame.xy))
        );
        surface candidate = surfaces[pixel.y * dimensions_frame.x + pixel.x];
        if (candidate.position.w < 0.5) continue;
        vec3 delta = candidate.position.xyz - position;
        float along = dot(delta, direction);
        float perpendicular = length(delta - direction * along);
        float thickness = max(environment_base_spacing.w * 0.08, distance * 0.035);
        if (along > 0.01 &&
            abs(along - distance) < thickness &&
            perpendicular < thickness) {
            return vec4(candidate.direct_emissive.xyz, 0.0);
        }
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
}

vec3 directional_c_minus_one_lod(vec3 position, vec3 normal, uint lod) {
    vec3 result = vec3(0.0);
    for (uint index = 0u; index < 32u; index++) {
        vec3 direction = direction_center(index, 0u);
        float cosine = max(0.0, dot(normal, direction));
        if (cosine <= 0.0) continue;
        vec4 near_interval = screen_space_near_interval(position, direction, lod);
        vec3 distant = sample_directional_lod(
            position,
            direction,
            0u,
            lod,
            true
        ).xyz;
        result += (near_interval.xyz + near_interval.w * distant) * cosine;
    }
    return result * (4.0 * PI / 32.0);
}

layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, dimensions_frame.xy))) return;
    uint pixel_index = pixel.y * dimensions_frame.x + pixel.x;
    surface surf = surfaces[pixel_index];
    vec3 color = surf.direct_emissive.xyz;
    vec3 indirect = vec3(0.0);
    vec3 display_history_sum = vec3(0.0);
    float display_confidence_sum = 0.0;
    float display_history_count = 0.0;
    if (surf.position.w > 0.5) {
        atomicAdd(state_words[diagnostics_offset() + 7u].value, 1u);
        float distance = max(max(abs(surf.position.x - camera_time.x), abs(surf.position.y - camera_time.y)), abs(surf.position.z - camera_time.z));
        float normalized = max(1.0, distance / (environment_base_spacing.w * 16.0));
        uint fine = uint(clamp(floor(log2(normalized)), 0.0, 7.0));
        uint coarse = min(7u, fine + 1u);
        float boundary = environment_base_spacing.w * 16.0 * exp2(float(fine + 1u));
        float blend = coarse == fine ? 0.0 : clamp((distance - boundary * 0.9) / (boundary * 0.1), 0.0, 1.0);
        vec3 normal = normalize(surf.normal_depth.xyz);
        vec3 irradiance = pass_params.z != 0u
            ? mix(
                directional_c_minus_one_lod(surf.position.xyz, normal, fine),
                directional_c_minus_one_lod(surf.position.xyz, normal, coarse),
                blend
            )
            : mix(
                sample_irradiance_lod(surf.position.xyz, normal, fine),
                sample_irradiance_lod(surf.position.xyz, normal, coarse),
                blend
            );
        vec3 view = normalize(camera_time.xyz - surf.position.xyz);
        indirect = surf.albedo.xyz / PI * irradiance;
        if (feature_flags.y != 0u) {
            indirect += mix(
                sample_rough_specular(surf.position.xyz, normal, view, fine),
                sample_rough_specular(surf.position.xyz, normal, view, coarse),
                blend
            );
        }
        if (dimensions_frame.z > 0u && pass_params.y == 0u) {
            vec4 previous_clip =
                previous_view_projection * vec4(surf.position.xyz, 1.0);
            if (previous_clip.w > 0.0) {
                vec2 previous_uv =
                    previous_clip.xy / previous_clip.w * 0.5 + vec2(0.5);
                if (all(greaterThanEqual(previous_uv, vec2(0.0))) &&
                    all(lessThanEqual(previous_uv, vec2(1.0)))) {
                    atomicAdd(state_words[diagnostics_offset() + 8u].value, 1u);
                    ivec2 previous_pixel = ivec2(min(
                        dimensions_frame.xy - uvec2(1u),
                        uvec2(previous_uv * vec2(dimensions_frame.xy))
                    ));
                    float position_tolerance =
                        max(0.02, environment_base_spacing.w * 0.12);
                    vec3 history_sum = vec3(0.0);
                    float history_count = 0.0;
                    for (int offset_y = -1; offset_y <= 1; offset_y++) {
                        for (int offset_x = -1; offset_x <= 1; offset_x++) {
                            ivec2 candidate_pixel = clamp(
                                previous_pixel + ivec2(offset_x, offset_y),
                                ivec2(0),
                                ivec2(dimensions_frame.xy) - ivec2(1)
                            );
                            uint previous_index =
                                uint(candidate_pixel.y) * dimensions_frame.x +
                                uint(candidate_pixel.x);
                            uint previous_base = presentation_history_base(
                                previous_frame(),
                                previous_index
                            );
                            vec4 previous_position =
                                load_state_vec4(previous_base);
                            vec4 previous_normal =
                                load_state_vec4(previous_base + 4u);
                            vec3 previous_indirect =
                                load_state_vec4(previous_base + 8u).xyz;
                            bool valid_sample =
                                previous_position.w > 0.5 &&
                                length(
                                    previous_position.xyz - surf.position.xyz
                                ) < position_tolerance &&
                                dot(
                                    normalize(previous_normal.xyz),
                                    normal
                                ) > 0.99 &&
                                !any(isnan(previous_indirect)) &&
                                !any(isinf(previous_indirect));
                            if (valid_sample) {
                                history_sum += previous_indirect;
                                history_count += 1.0;
                                vec4 previous_display =
                                    load_state_vec4(previous_base + 12u);
                                if (!any(isnan(previous_display)) &&
                                    !any(isinf(previous_display))) {
                                    display_history_sum += previous_display.xyz;
                                    display_confidence_sum +=
                                        max(1.0, previous_display.w);
                                    display_history_count += 1.0;
                                }
                            }
                        }
                    }
                    if (history_count > 0.0) {
                        atomicAdd(state_words[diagnostics_offset() + 6u].value, 1u);
                        float presentation_weight =
                            0.0;
                        indirect = mix(
                            indirect,
                            history_sum / history_count,
                            presentation_weight
                        );
                    }
                }
            }
        }
        color += indirect;
    }
    uint history_base = presentation_history_base(
        current_frame(),
        pixel_index
    );
    store_state_vec4(history_base, surf.position);
    store_state_vec4(
        history_base + 4u,
        surf.position.w > 0.5
            ? vec4(normalize(surf.normal_depth.xyz), 1.0)
            : vec4(0.0)
    );
    store_state_vec4(history_base + 8u, vec4(indirect, 1.0));
    color *= point_color_exposure.w;
    color = color / (vec3(1.0) + color);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    float display_confidence = 1.0;
    if (pass_params.y == 0u && display_history_count > 0.0) {
        display_confidence = min(
            4096.0,
            display_confidence_sum / display_history_count + 1.0
        );
        color = mix(
            color,
            display_history_sum / display_history_count,
            0.5
        );
    }
    if (display_confidence > 100.0) {
        atomicAdd(state_words[diagnostics_offset() + 11u].value, 1u);
    }
    store_state_vec4(
        history_base + 12u,
        vec4(color, display_confidence)
    );
    imageStore(output_image, ivec2(pixel), vec4(color, 1.0));
}
@end
@program shade cs_shade

@vs vs_fullscreen
out vec2 uv;
void main() {
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs fs_present
layout(binding=0) uniform texture2D display_texture;
layout(binding=0) uniform sampler display_sampler;
in vec2 uv;
out vec4 frag_color;
void main() {
    frag_color = texture(sampler2D(display_texture, display_sampler), uv);
}
@end
@program present vs_fullscreen fs_present
