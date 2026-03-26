"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { Suspense } from "react";
import type { GalaxyElephant, Position3D } from "../../../lib/types";
import GalaxyStars from "./GalaxyStars";
import UploadedPoint from "./UploadedPoint";

interface Props {
  elephants: GalaxyElephant[];
  uploadedPosition?: Position3D | null;
  nearestPosition?: Position3D | null;
  uploadedImageUrl?: string | null;
}

export default function GalaxyScene({
  elephants,
  uploadedPosition,
  nearestPosition,
  uploadedImageUrl,
}: Props) {
  return (
    <Canvas
      camera={{ position: [0, 40, 80], fov: 60 }}
      style={{ background: "#050510" }}
      dpr={[1, 2]}
    >
      <ambientLight intensity={0.25} />
      <pointLight position={[0, 5, 0]} intensity={3} color="#00d4aa" distance={250} decay={1.5} />
      <pointLight position={[40, 30, -40]} intensity={1.2} color="#7b61ff" distance={180} decay={1.5} />
      <pointLight position={[-20, -15, 30]} intensity={0.6} color="#f59e42" distance={120} decay={2} />
      <directionalLight position={[50, 60, 30]} intensity={0.8} color="#ffffff" />

      <Suspense fallback={null}>
        <GalaxyStars elephants={elephants} />
        {uploadedPosition && (
          <UploadedPoint position={uploadedPosition} nearestPosition={nearestPosition ?? null} imageUrl={uploadedImageUrl} />
        )}
      </Suspense>

      <Stars radius={300} depth={80} count={5000} factor={3} saturation={0.3} fade speed={0.4} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        autoRotate
        autoRotateSpeed={0.2}
        maxDistance={180}
        minDistance={15}
        enablePan={false}
      />
    </Canvas>
  );
}
