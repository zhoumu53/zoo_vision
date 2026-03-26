"use client";

import { useRef, useState, useMemo } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { Html, Billboard } from "@react-three/drei";
import * as THREE from "three";
import type { GalaxyElephant } from "../../../lib/types";
import { ELEPHANTS as ELEPHANT_INFO } from "../../../lib/elephants";

function spiralPosition(index: number, total: number): [number, number, number] {
  const goldenAngle = 2.39996323;
  const angle = index * goldenAngle;
  const r = 3 + 0.6 * Math.sqrt(index + 1) * Math.sqrt(total / 10);
  const x = r * Math.cos(angle);
  const z = r * Math.sin(angle);
  const y = Math.sin(index * 1.7) * 2 + Math.cos(index * 0.9) * 1.5;
  return [x, y, z];
}

function elephantPosition(el: GalaxyElephant, index: number, total: number): [number, number, number] {
  if (el.x !== null && el.y !== null && el.z !== null) return [el.x, el.y, el.z];
  return spiralPosition(index, total);
}

/** Create a circular profile texture from a loaded image. */
function createCircularProfileTexture(image: HTMLImageElement, borderColor: string, size: number = 512): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);

  const r = size * 0.44;
  const cx = size / 2;
  const cy = size / 2;

  // Clip to circle and draw profile photo
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();

  // Draw image covering the circle (center-crop)
  const imgAspect = image.width / image.height;
  let sx = 0, sy = 0, sw = image.width, sh = image.height;
  if (imgAspect > 1) {
    sw = image.height;
    sx = (image.width - sw) / 2;
  } else {
    sh = image.width;
    sy = (image.height - sh) / 2;
  }
  ctx.drawImage(image, sx, sy, sw, sh, cx - r, cy - r, r * 2, r * 2);
  ctx.restore();

  // Subtle vignette overlay
  const vignette = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r);
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(0.7, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,0.35)");
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = vignette;
  ctx.fill();

  // Colored border ring
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = size * 0.025;
  ctx.stroke();

  // Outer glow
  ctx.beginPath();
  ctx.arc(cx, cy, r + size * 0.015, 0, Math.PI * 2);
  ctx.strokeStyle = borderColor.replace(")", ", 0.3)").replace("rgb", "rgba");
  ctx.lineWidth = size * 0.02;
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createGlowTexture(hue: number, size: number = 256): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, `hsla(${hue}, 65%, 50%, 0.6)`);
  grad.addColorStop(0.3, `hsla(${hue}, 65%, 50%, 0.2)`);
  grad.addColorStop(0.7, `hsla(${hue}, 65%, 50%, 0.05)`);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Convert hex color to CSS rgb string */
function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${r}, ${g}, ${b})`;
}

function ElephantAvatar({
  elephant, index, total,
}: {
  elephant: GalaxyElephant;
  index: number;
  total: number;
}) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hovered, setHovered] = useState(false);
  const [profileTexture, setProfileTexture] = useState<THREE.CanvasTexture | null>(null);

  const pos = useMemo(() => elephantPosition(elephant, index, total), [elephant, index, total]);
  const radius = useMemo(() => Math.log(elephant.image_count + 1) * 0.3 + 0.5, [elephant.image_count]);

  const info = ELEPHANT_INFO.find((e) => e.name === elephant.elephant_name);
  const color = info?.color || "#ffffff";
  const baseHue = useMemo(() => info ? [55, 210, 270, 15, 130, 30][ELEPHANT_INFO.indexOf(info)] || (index / total) * 360 : (index / total) * 360, [info, index, total]);

  const glowTexture = useMemo(() => createGlowTexture(baseHue), [baseHue]);

  // Load profile image and create circular texture
  const profilePath = elephant.sample_crop_path || (elephant as any).profile;
  useMemo(() => {
    if (!profilePath) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const tex = createCircularProfileTexture(img, hexToRgb(color));
      setProfileTexture(tex);
    };
    img.src = profilePath;
  }, [profilePath, color]);

  useFrame((_, delta) => {
    if (groupRef.current) {
      const target = hovered ? 1.2 : 1;
      const s = groupRef.current.scale.x;
      groupRef.current.scale.setScalar(THREE.MathUtils.lerp(s, target, delta * 6));
    }
  });

  const ballSize = radius * 2.5;

  return (
    <group position={pos} ref={groupRef}>
      {/* Profile photo ball */}
      <Billboard follow lockX={false} lockY={false} lockZ={false}>
        <mesh
          onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = "pointer"; }}
          onPointerOut={() => { setHovered(false); document.body.style.cursor = "default"; }}
        >
          <planeGeometry args={[ballSize, ballSize]} />
          {profileTexture ? (
            <meshBasicMaterial map={profileTexture} transparent depthWrite={false} toneMapped={false} />
          ) : (
            <meshBasicMaterial color={color} transparent opacity={0.3} depthWrite={false} />
          )}
        </mesh>
      </Billboard>

      {/* Glow behind the ball */}
      <Billboard>
        <mesh>
          <planeGeometry args={[ballSize * 1.4, ballSize * 1.4]} />
          <meshBasicMaterial map={glowTexture} transparent opacity={hovered ? 0.5 : 0.2} depthWrite={false} toneMapped={false} />
        </mesh>
      </Billboard>

      {/* Name label always visible above the ball */}
      <Html
        center
        distanceFactor={60}
        position={[0, ballSize * 0.65, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div className="whitespace-nowrap text-center">
          <div
            className="text-xs font-semibold tracking-wide drop-shadow-[0_1px_4px_rgba(0,0,0,0.8)]"
            style={{ color }}
          >
            {elephant.elephant_name}
          </div>
        </div>
      </Html>

      {/* Hover tooltip with extra info */}
      {hovered && (
        <Html
          center
          distanceFactor={50}
          position={[0, -ballSize * 0.7, 0]}
          style={{ pointerEvents: "none" }}
        >
          <div className="whitespace-nowrap rounded-xl border border-white/10 bg-black/80 px-4 py-2 text-center backdrop-blur-md">
            <div className="text-sm font-medium" style={{ color }}>{elephant.elephant_name}</div>
            <div className="text-xs text-white/50">{elephant.image_count.toLocaleString()} photos</div>
          </div>
        </Html>
      )}
    </group>
  );
}

export default function GalaxyStars({ elephants }: { elephants: GalaxyElephant[] }) {
  const groupRef = useRef<THREE.Group>(null!);
  return (
    <group ref={groupRef}>
      {elephants.map((el, i) => (
        <ElephantAvatar key={el.elephant_id} elephant={el} index={i} total={elephants.length} />
      ))}
    </group>
  );
}
