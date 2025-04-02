#include <iostream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <random>
#include <vector>
#include <ctime>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int width = 512;
const int height = 512;
const int maxDepth = 8;

struct Vec3 {
    float x, y, z;

    Vec3 operator+(const Vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    Vec3 operator-(const Vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
    Vec3 operator-() const { return { -x, -y, -z }; }
    Vec3 operator*(float f) const { return { x * f, y * f, z * f }; }
    Vec3 operator*(const Vec3& v) const { return { x * v.x, y * v.y, z * v.z }; }
    Vec3 operator/(float f) const { return { x / f, y / f, z / f }; }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    float length() const { return std::sqrt(dot(*this)); }
    Vec3 normalize() const {
        float len = length();
        return *this / (len + 1e-6f);
    }
    Vec3 cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }
};

using Color = Vec3;

enum MaterialType {
    DIFFUSE,
    METAL,
    GLASS,
    LIGHT
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    float sadness = 0.1f;
};

struct Sphere {
    Vec3 center;
    float radius;
    Color baseColor;
    float emotionalDissonance = 0.4f;
    MaterialType material = DIFFUSE;
    float roughness = 0.0f;
    float ior = 1.5f;
    float absorption = 0.1f;
    bool isLight = false;
    
    bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float b = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float h = b * b - c;
        if (h < 0.0f) return false;
        h = std::sqrt(h);
        t = -b - h;
        if (t < 0.001f) t = -b + h;
        return t > 0.001f;
    }
};

struct Camera {
    Vec3 position;
    Vec3 lookAt;
    Vec3 up;
    float fov;
    float aspectRatio;
    
    Camera(Vec3 pos, Vec3 look, Vec3 u, float f, float aspect)
        : position(pos), lookAt(look), up(u), fov(f), aspectRatio(aspect) {
    }
    
    Ray getRay(float u, float v) {
        float theta = fov * M_PI / 180.0f;
        float halfHeight = tan(theta / 2.0f);
        float halfWidth = aspectRatio * halfHeight;
        
        Vec3 w = (position - lookAt).normalize();
        Vec3 u_vec = up.cross(w).normalize();
        Vec3 v_vec = w.cross(u_vec);
        
        Vec3 lowerLeftCorner = position - u_vec * halfWidth - v_vec * halfHeight - w;
        Vec3 horizontal = u_vec * 2.0f * halfWidth;
        Vec3 vertical = v_vec * 2.0f * halfHeight;
        
        Ray r;
        r.origin = position;
        r.direction = (lowerLeftCorner + horizontal * u + vertical * v - position).normalize();
        r.sadness = 0.1f;
        return r;
    }
};

std::vector<Sphere> scene = {
    {{-1.0f, 0.0f, -3.0f}, 0.7f, {0.98f, 0.99f, 0.99f}, 0.3f, GLASS, 0.01f, 1.5f, 0.02f, false},
    {{1.0f, 0.0f, -3.0f}, 0.7f, {0.7f, 0.7f, 0.8f}, 0.7f, METAL, 0.05f, 0.0f, 0.0f, false},
    {{0.0f, -1000.7f, -3.0f}, 1000.0f, {0.2f, 0.2f, 0.3f}, 0.2f, DIFFUSE, 0.0f, 0.0f, 0.0f, false},
    {{-3.0f, 3.0f, -6.0f}, 1.0f, {0.9f, 0.9f, 1.0f}, 0.0f, DIFFUSE, 0.0f, 0.0f, 0.0f, true}
};

std::mt19937 rng;
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

Vec3 randomCosineHemisphere(const Vec3& normal) {
    float u1 = dist(rng);
    float u2 = dist(rng);

    float r = std::sqrt(u1);
    float theta = 2.0f * M_PI * u2;

    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(1.0f - u1);

    Vec3 w = normal;
    Vec3 a = (std::abs(w.x) > 0.1f) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 u = (a - w * w.dot(a)).normalize();
    Vec3 v = w.cross(u);

    return (u * x + v * y + w * z).normalize();
}

Vec3 randomInUnitSphere() {
    while (true) {
        Vec3 p = {dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f};
        if (p.dot(p) < 1.0f)
            return p;
    }
}

Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * 2.0f * v.dot(n);
}

bool refract(const Vec3& v, const Vec3& n, float niOverNt, Vec3& refracted) {
    Vec3 uv = v.normalize();
    float dt = uv.dot(n);
    float discriminant = 1.0f - niOverNt * niOverNt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = (uv - n * dt) * niOverNt - n * std::sqrt(discriminant);
        return true;
    }
    return false;
}

float fresnelSchlick(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}

float ggxDistribution(const Vec3& n, const Vec3& h, float roughness) {
    float alpha2 = roughness * roughness;
    float NdotH = std::max(n.dot(h), 0.0f);
    float NdotH2 = NdotH * NdotH;
    
    float denom = NdotH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float k = roughness * roughness / 2.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

float geometrySmith(const Vec3& n, const Vec3& v, const Vec3& l, float roughness) {
    float NdotV = std::max(n.dot(v), 0.0f);
    float NdotL = std::max(n.dot(l), 0.0f);
    float ggx1 = geometrySchlickGGX(NdotV, roughness);
    float ggx2 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

Vec3 emotionalLightDir = Vec3{-1, -1, -1}.normalize();
Color emotionalLightColor = {0.8f, 0.8f, 1.0f};

bool directLightSample(const Vec3& origin, Vec3& direction, Color& lightColor, float& distance, float& sadness) {
    if (dist(rng) < 0.3f) {
        return false;
    }
    
    std::vector<const Sphere*> lights;
    for (const auto& s : scene) {
        if (s.isLight) {
            lights.push_back(&s);
        }
    }
    
    if (lights.empty()) return false;
    
    const Sphere* light = lights[0];
    
    float u1 = dist(rng);
    float u2 = dist(rng);
    
    float theta = 2.0f * M_PI * u1;
    float phi = std::acos(2.0f * u2 - 1.0f);
    
    Vec3 spherePoint = {
        std::sin(phi) * std::cos(theta),
        std::sin(phi) * std::sin(theta),
        std::cos(phi)
    };
    
    Vec3 lightPoint = light->center + spherePoint * light->radius;
    
    direction = (lightPoint - origin).normalize();
    distance = (lightPoint - origin).length();
    distance = distance * distance;
    lightColor = light->baseColor;
    
    sadness = 0.15f;
    
    for (const auto& s : scene) {
        if (!s.isLight) {
            float t = 0;
            Ray shadowRay;
            shadowRay.origin = origin;
            shadowRay.direction = direction;
            
            if (s.intersect(shadowRay, t) && t * t < distance) {
                return false;
            }
        }
    }
    
    return true;
}

Color trace(const Ray& ray, int depth) {
    if (depth > maxDepth || ray.sadness > 1.0f)
        return { 0.1f, 0.1f, 0.2f };

    float tMin = std::numeric_limits<float>::max();
    const Sphere* hitSphere = nullptr;
    float tHit = 0;

    for (const auto& s : scene) {
        float t = 0;
        if (s.intersect(ray, t) && t < tMin) {
            tMin = t;
            tHit = t;
            hitSphere = &s;
        }
    }

    if (!hitSphere) return { 0.05f, 0.05f, 0.1f };

    if (hitSphere->isLight) {
        return hitSphere->baseColor;
    }

    Vec3 hitPoint = ray.origin + ray.direction * tHit;
    Vec3 normal = (hitPoint - hitSphere->center).normalize();
    Vec3 normal_facing = normal.dot(ray.direction) < 0 ? normal : normal * -1.0f;
    Vec3 view_dir = -ray.direction;

    float lightIntensity = std::max(0.0f, normal.dot(-emotionalLightDir));
    Color emotionalImpact = emotionalLightColor * lightIntensity * 0.3f;
    
    if (hitSphere->material == METAL) {
        Vec3 reflected = reflect(ray.direction.normalize(), normal_facing);
        
        if (hitSphere->roughness > 0.0f) {
            reflected = (reflected + randomInUnitSphere() * hitSphere->roughness).normalize();
        }
        
        Ray reflectedRay;
        reflectedRay.origin = hitPoint + normal_facing * 0.001f;
        reflectedRay.direction = reflected;
        reflectedRay.sadness = ray.sadness + 0.1f * hitSphere->emotionalDissonance;
        
        Color reflectedColor = trace(reflectedRay, depth + 1);
        return (hitSphere->baseColor * reflectedColor) + emotionalImpact * 0.2f;
    }
    else if (hitSphere->material == GLASS) {
        Vec3 outwardNormal;
        Vec3 reflected = reflect(ray.direction, normal_facing);
        float niOverNt;
        float cosine;
        bool isEntering = ray.direction.dot(normal) < 0;
        
        if (!isEntering) {
            outwardNormal = -normal;
            niOverNt = hitSphere->ior;
            cosine = hitSphere->ior * ray.direction.dot(-normal) / ray.direction.length();
        } else {
            outwardNormal = normal;
            niOverNt = 1.0f / hitSphere->ior;
            cosine = -ray.direction.dot(normal) / ray.direction.length();
        }
        
        Vec3 refracted;
        float reflectProb;
        
        bool canRefract = refract(ray.direction, outwardNormal, niOverNt, refracted);
        if (canRefract) {
            reflectProb = fresnelSchlick(std::abs(cosine), hitSphere->ior);
        } else {
            reflectProb = 1.0f;
        }
        
        Color resultColor;
        if (dist(rng) < reflectProb) {
            Vec3 perturbedReflection = reflected;
            if (hitSphere->roughness > 0.0f) {
                perturbedReflection = (reflected + randomInUnitSphere() * hitSphere->roughness).normalize();
            }
            
            Ray reflectedRay;
            reflectedRay.origin = hitPoint + normal_facing * 0.001f;
            reflectedRay.direction = perturbedReflection;
            reflectedRay.sadness = ray.sadness + 0.05f * hitSphere->emotionalDissonance;
            
            resultColor = trace(reflectedRay, depth + 1);
        } else {
            Vec3 perturbedRefraction = refracted;
            if (hitSphere->roughness > 0.0f) {
                perturbedRefraction = (refracted + randomInUnitSphere() * hitSphere->roughness).normalize();
            }
            
            Ray refractedRay;
            refractedRay.origin = hitPoint - outwardNormal * 0.001f;
            refractedRay.direction = perturbedRefraction;
            refractedRay.sadness = ray.sadness * (1.0f - hitSphere->absorption);
            
            Color absorbance = {1.0f, 1.0f, 1.0f};
            if (!isEntering) {
                float distInside = tHit;
                absorbance.x = std::exp(-hitSphere->baseColor.x * hitSphere->absorption * distInside * 10.0f);
                absorbance.y = std::exp(-hitSphere->baseColor.y * hitSphere->absorption * distInside * 10.0f);
                absorbance.z = std::exp(-hitSphere->baseColor.z * hitSphere->absorption * distInside * 10.0f);
            }
            
            resultColor = trace(refractedRay, depth + 1) * absorbance;
        }
        
        return resultColor + emotionalImpact * 0.05f;
    }
    else {
        Color directLight = {0, 0, 0};
        Vec3 lightDir;
        Color lightColor;
        float lightDist;
        float lightSadness;
        
        if (directLightSample(hitPoint + normal_facing * 0.001f, lightDir, lightColor, lightDist, lightSadness)) {
            float ndotl = std::max(0.0f, normal_facing.dot(lightDir));
            directLight = lightColor * ndotl * (1.0f - lightSadness);
        }

        Ray nextRay;
        nextRay.origin = hitPoint + normal_facing * 0.001f;
        nextRay.direction = randomCosineHemisphere(normal_facing);
        nextRay.sadness = ray.sadness + 0.2f * hitSphere->emotionalDissonance;

        Color indirectLight = trace(nextRay, depth + 1);
        Color local = hitSphere->baseColor * (1.0f - ray.sadness);
        
        float moodBalance = 0.7f;
        Color result = (local * (directLight * moodBalance + indirectLight * (1.0f - moodBalance) + emotionalImpact));
        
        result.z += 0.05f * (1.0f - ray.sadness);
        
        return result;
    }
}

void render() {
    std::ofstream img("emotracer.ppm");
    img << "P3\n" << width << " " << height << "\n255\n";

    const int samplesPerPixel = 8;
    
    Camera camera(
        Vec3{0.0f, 0.5f, 1.0f},  
        Vec3{0.0f, 0.0f, -3.0f}, 
        Vec3{0.0f, 1.0f, 0.0f}, 
        60.0f,                  
        float(width) / float(height) 
    );

    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            Color pixelColor = {0, 0, 0};
            
            for (int s = 0; s < samplesPerPixel; ++s) {
                float u = (x + dist(rng)) / width;
                float v = (y + dist(rng)) / height;

                Ray r = camera.getRay(u, v);
                pixelColor = pixelColor + trace(r, 0);
            }
            
            pixelColor = pixelColor / static_cast<float>(samplesPerPixel);
            
            pixelColor.x = std::min(1.0f, pixelColor.x);
            pixelColor.y = std::min(1.0f, pixelColor.y);
            pixelColor.z = std::min(1.0f, pixelColor.z);
            
            pixelColor.x = std::pow(pixelColor.x, 1.0f/2.2f);
            pixelColor.y = std::pow(pixelColor.y, 1.0f/2.2f);
            pixelColor.z = std::pow(pixelColor.z, 1.0f/2.2f);

            int r8 = std::min(255, int(255.99f * pixelColor.x));
            int g8 = std::min(255, int(255.99f * pixelColor.y));
            int b8 = std::min(255, int(255.99f * pixelColor.z));
            img << r8 << " " << g8 << " " << b8 << "\n";
        }
    }

    img.close();
    std::cout << "emotracer.ppm written.\n";
}

int main() {
    rng.seed(static_cast<unsigned int>(time(0)));
    render();
    return 0;
}