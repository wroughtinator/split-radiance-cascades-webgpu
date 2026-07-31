import type { Metadata } from "next";
import { RadianceCascadesLab } from "./radiance-cascades-lab";

export const metadata: Metadata = {
  title: "Split Radiance Cascades — WebGPU GI Lab",
  description:
    "A production WebGPU implementation of Split Radiance Cascades with twelve real-time validation scenes.",
};

export default function Home() {
  return <RadianceCascadesLab />;
}
