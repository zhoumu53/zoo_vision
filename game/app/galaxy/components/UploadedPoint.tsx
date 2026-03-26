"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html, Billboard, Line } from "@react-three/drei";
import * as THREE from "three";
import type { Position3D } from "../../../lib/types";

function createFallbackOrangeTexture(size: number = 512): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  const grad = ctx.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, "hsl(20, 80%, 40%)");
  grad.addColorStop(1, "hsl(30, 90%, 30%)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const highlightGrad = ctx.createRadialGradient(size * 0.38, size * 0.35, 0, size * 0.42, size * 0.40, size * 0.28);
  highlightGrad.addColorStop(0, "rgba(255,255,255,0.35)");
  highlightGrad.addColorStop(0.3, "rgba(255,255,255,0.12)");
  highlightGrad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
  ctx.fillStyle = highlightGrad;
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createOrangeGlowTexture(size: number = 256): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "hsla(20, 90%, 50%, 0.5)");
  grad.addColorStop(0.3, "hsla(20, 90%, 50%, 0.15)");
  grad.addColorStop(0.7, "hsla(20, 90%, 50%, 0.03)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

interface Props {
  position: Position3D;
  nearestPosition: Position3D | null;
  imageUrl?: string | null;
}

export default function UploadedPoint({ position, nearestPosition }: Props) {
  const glowRef = useRef<THREE.Sprite>(null!);
  const fallbackTexture = useMemo(() => createFallbackOrangeTexture(), []);
  const glowTexture = useMemo(() => createOrangeGlowTexture(), []);

  const pos: [number, number, number] = [position.x, position.y, position.z];
  const planetRadius = 2.0;

  useFrame((state) => {
    if (glowRef.current) {
      const opacity = 0.7 + Math.sin(state.clock.elapsedTime * 2) * 0.3;
      (glowRef.current.material as THREE.SpriteMaterial).opacity = opacity;
    }
  });

  return (
    <group>
      <sprite ref={glowRef} position={pos} scale={[planetRadius * 5, planetRadius * 5, 1]}>
        <spriteMaterial map={glowTexture} transparent opacity={0.7} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
      <Billboard position={pos}>
        <mesh>
          <planeGeometry args={[planetRadius * 2, planetRadius * 2]} />
          <meshBasicMaterial map={fallbackTexture} transparent depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      </Billboard>
      <pointLight position={pos} color="#ff6b35" intensity={2} distance={20} decay={2} />
      <Html position={[pos[0], pos[1] + planetRadius + 1.5, pos[2]]} center distanceFactor={50} style={{ pointerEvents: "none" }}>
        <div className="flex flex-col items-center gap-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-orange-500 text-base font-bold text-white shadow-lg shadow-orange-500/50">!</div>
          <div className="whitespace-nowrap rounded-xl border border-orange-400/30 bg-black/80 px-3 py-1.5 text-center backdrop-blur-md">
            <div className="text-xs font-medium text-orange-300">Your Photo</div>
          </div>
        </div>
      </Html>
      {nearestPosition && (
        <Line
          points={[[position.x, position.y, position.z], [nearestPosition.x, nearestPosition.y, nearestPosition.z]]}
          color="#ff6b35"
          lineWidth={1.5}
          transparent
          opacity={0.4}
        />
      )}
    </group>
  );
}
