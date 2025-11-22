import { translate, type Language } from "../i18n";

export const INFERENCE_CLASS_DESCRIPTION_TEXT = translate("ja", "inference.classText");

export const getInferenceClassDescription = (index: number, language: Language = "ja"): string => {
  const key = `inference.class.${index}`;
  return translate(language, key);
};

export const getInferenceClassDescriptionText = (language: Language = "ja"): string =>
  translate(language, "inference.classText");
