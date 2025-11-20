export const INFERENCE_CLASS_DESCRIPTIONS = [
  "単一細胞",
  "複数細胞",
  "ピンぼけ",
  "非細胞粒子",
] as const;

export const INFERENCE_CLASS_DESCRIPTION_TEXT =
  "0＝単一細胞、1＝複数細胞、2＝ピンぼけ、3＝非細胞粒子";

export const getInferenceClassDescription = (index: number): string => {
  const description = INFERENCE_CLASS_DESCRIPTIONS[index];
  return description ?? "";
};
