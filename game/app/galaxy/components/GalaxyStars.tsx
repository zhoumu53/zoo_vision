"use client";

import { useRef, useState, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
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

function createFallbackCircleTexture(seed: number, baseHue: number, size: number = 512): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);

  ctx.save();
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();

  const grad = ctx.createLinearGradient(0, 0, size, size);
  const h2 = (baseHue + 40) % 360;
  grad.addColorStop(0, `hsl(${baseHue}, 55%, 30%)`);
  grad.addColorStop(1, `hsl(${h2}, 60%, 20%)`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  ctx.restore();

  const highlightGrad = ctx.createRadialGradient(size * 0.38, size * 0.35, 0, size * 0.42, size * 0.40, size * 0.28);
  highlightGrad.addColorStop(0, "rgba(255,255,255,0.35)");
  highlightGrad.addColorStop(0.3, "rgba(255,255,255,0.12)");
  highlightGrad.addColorStop(0.7, "rgba(255,255,255,0)");
  highlightGrad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.fillStyle = highlightGrad;
  ctx.fill();

  const edgeGrad = ctx.createRadialGradient(size / 2, size / 2, size * 0.25, size / 2, size / 2, size * 0.42);
  edgeGrad.addColorStop(0, "rgba(0,0,0,0)");
  edgeGrad.addColorStop(0.6, "rgba(0,0,0,0)");
  edgeGrad.addColorStop(0.85, "rgba(0,0,0,0.15)");
  edgeGrad.addColorStop(1, "rgba(0,0,0,0.4)");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.fillStyle = edgeGrad;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.strokeStyle = `hsla(${baseHue}, 65%, 50%, 0.2)`;
  ctx.lineWidth = size * 0.005;
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

function ElephantAvatar({
  elephant, index, total,
}: {
  elephant: GalaxyElephant;
  index: number;
  total: number;
}) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hovered, setHovered] = useState(false);

  const pos = useMemo(() => elephantPosition(elephant, index, total), [elephant, index, total]);
  const radius = useMemo(() => Math.log(elephant.image_count + 1) * 0.3 + 0.5, [elephant.image_count]);

  const info = ELEPHANT_INFO.find((e) => e.name === elephant.elephant_name);
  const baseHue = useMemo(() => info ? [55, 210, 270, 15, 130][ELEPHANT_INFO.indexOf(info)] || (index / total) * 360 : (index / total) * 360, [info, index, total]);

  const circleTexture = useMemo(() => createFallbackCircleTexture(elephant.elephant_id * 137, baseHue), [elephant.elephant_id, baseHue]);
  const glowTexture = useMemo(() => createGlowTexture(baseHue), [baseHue]);

  useFrame((_, delta) => {
    if (groupRef.current) {
      const target = hovered ? 1.2 : 1;
      const s = groupRef.current.scale.x;
      groupRef.current.scale.setScalar(THREE.MathUtils.lerp(s, target, delta * 6));
    }
  });

  return (
    <group position={pos} ref={groupRef}>
      <Billboard follow lockX={false} lockY={false} lockZ={false}>
        <mesh
          onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = "pointer"; }}
          onPointerOut={() => { setHovered(false); document.body.style.cursor = "default"; }}
        >
          <planeGeometry args={[radius * 2.2, radius * 2.2]} />
          <meshBasicMaterial map={circleTexture} transparent depthWrite={false} toneMapped={false} />
        </mesh>
      </Billboard>
      <Billboard>
        <mesh>
          <planeGeometry args={[radius * 3, radius * 3]} />
          <meshBasicMaterial map={glowTexture} transparent opacity={hovered ? 0.4 : 0.15} depthWrite={false} toneMapped={false} />
        </mesh>
      </Billboard>
      {hovered && (
        <Html center distanceFactor={50} style={{ pointerEvents: "none" }}>
          <div className="whitespace-nowrap rounded-xl border border-white/10 bg-black/80 px-4 py-2 text-center backdrop-blur-md">
            <div className="text-sm font-medium" style={{ color: info?.color || "#fff" }}>{elephant.elephant_name}</div>
            <div className="text-xs text-white/50">{elephant.image_count} photos</div>
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
