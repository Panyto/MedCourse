import json
import os
import random
import re
from collections import Counter, defaultdict
from json import JSONDecodeError

from vllm import LLM, SamplingParams


class LLMCARe:
    def __init__(self, model_path, ref_path, mapping_path):
        self.fields = [
            "chief_complaint",
            "history_of_present_illness",
            "hospital_course",
            "discharge_instructions",
        ]
        self._load_ref(ref_path)
        self._load_model(model_path)
        with open(mapping_path) as f:
            self.icd2diag = json.load(f)
        self._initialize_template()

    def _load_model(self, model_path):
        self.llm = LLM(
            model=model_path,
            max_model_len=8192,
        )
        with open(os.path.join(model_path, "generation_config.json")) as f:
            generation_config = json.load(f)
        self.default_sampling_params = dict(max_tokens=4096)
        for i in generation_config:
            if i in [
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "temperature",
                "top_p",
                "top_k",
            ]:
                self.default_sampling_params[i] = generation_config[i]

    def _generate(self, prompt, sampling_params=dict()):
        sampling_params = self.default_sampling_params | sampling_params
        if isinstance(prompt, str):
            message = [{"role": "user", "content": prompt}]
        else:
            message = [[{"role": "user", "content": i}] for i in prompt]
        output = self.llm.chat(
            message,
            sampling_params=SamplingParams(**sampling_params),
        )
        output = [j.text for i in output for j in i.outputs]
        return output

    def _initialize_template(self):
        self.template = {
            "generator": "Please generate an electronic medical record according to the requirements below:\n"
            + "1. The primary discharge diagnosis must be: {condition}.\n"
            + '2. The record must contain exactly the following fields, without adding or removing any: "patient_info, chief_complaint, history_of_present_illness, hospital_course, discharge_diagnosis, discharge_instructions".\n'
            + '3. Field requirements: "patient_info" must include sex and age; "chief_complaint" should briefly describe the reason for admission; "history_of_present_illness" should detail disease progression and current status; "hospital_course" should include examinations and treatments; "discharge_instructions" should include medications.\n'
            + '4. The final output must be in JSON format as "field_name": "field_content", and each field content must be a single string.',
            "critic": "Below is the {field} section of an electronic medical record:\n"
            + "```json\n{emr}\n```\n"
            + 'Please determine whether this {field} satisfies the requirement: "{requirement}".\n'
            + "Return the result in the following JSON format:\n"
            + "```json\n{{\n"
            + '    "meets_requirement": true/false\n'
            + "}}\n```",
            "adviser": "Below is the {field} section of an electronic medical record:\n"
            + "```json\n{emr}\n```\n"
            + 'This content violates the requirement: "{requirement}". Please provide a revision suggestion. If multiple fields are involved, choose the single field with minimal impact to modify.\n'
            + "Return the result in the following JSON format:\n"
            + "```json\n{{\n"
            + '    "field_to_modify": "patient_info/chief_complaint/history_of_present_illness/hospital_course/discharge_diagnosis/discharge_instructions",\n'
            + '    "modification_suggestion": "specific revision suggestion"\n'
            + "}}\n```\n"
            + "In the suggestion, describe how to revise based on the field content only. Do not directly output revised EMR text.",
            "reviser": "Below is an electronic medical record containing errors:\n"
            + "```json\n{emr}\n```\n"
            + "The {field} field has issues. Please revise this field according to the feedback: {feedback}\n"
            + 'Please output the result in JSON format, keeping the format as: {{"patient_info/chief_complaint/history_of_present_illness/hospital_course/discharge_diagnosis/discharge_instructions": "field_content"}}.',
        }
        self.criteria = {
            "completeness": [
                (
                    "chief_complaint",
                    "Includes at least one of the following: symptom, examination result, or disease name; it is not necessary to include all three.",
                ),
                (
                    "chief_complaint",
                    "Includes the onset time of the main symptom or sign.",
                ),
                (
                    "history_of_present_illness",
                    "Mentions the onset pattern, such as sudden onset or chronic progression.",
                ),
                (
                    "history_of_present_illness",
                    "States whether there is a cause or trigger; wording like 'no obvious trigger' also satisfies this requirement, or a specific cause can be provided.",
                ),
                (
                    "history_of_present_illness",
                    "Includes the main symptoms or signs.",
                ),
                (
                    "history_of_present_illness",
                    "Explicitly lists all of the following general conditions: appetite/diet, bowel and urination, mental status, sleep, and weight change.",
                ),
                (
                    "hospital_course",
                    "Includes examinations or laboratory tests performed for the patient.",
                ),
                (
                    "hospital_course",
                    "Includes treatments received by the patient.",
                ),
                (
                    "discharge_instructions",
                    "Provides dosage and administration instructions for each medication mentioned.",
                ),
            ],
            "consistency": [
                (
                    ("chief_complaint", "history_of_present_illness"),
                    "Checks whether all symptoms in the chief complaint are covered in the history of present illness; synonymous wording is acceptable and exact phrasing is not required.",
                ),
                (
                    ("chief_complaint", "history_of_present_illness"),
                    "Checks whether symptom duration is consistent between the chief complaint and the history of present illness; for each symptom with a stated duration in the chief complaint, the history should state a matching duration.",
                ),
                (
                    ("history_of_present_illness", "hospital_course"),
                    "Checks whether the affected body part described in the history of present illness is consistent with the body part examined and treated in the hospital course.",
                ),
            ],
            "correctness": [
                (
                    ("discharge_diagnosis", "patient_info"),
                    "The discharge diagnosis should not conflict with the patient's sex or age.",
                ),
                (
                    ("discharge_diagnosis", "chief_complaint"),
                    "The discharge diagnosis should not conflict with symptoms in the chief complaint; the chief complaint should not contain symptoms that are clearly inconsistent with the discharge diagnosis.",
                ),
                (
                    ("discharge_diagnosis", "history_of_present_illness"),
                    "The discharge diagnosis should not conflict with symptoms in the history of present illness; the history should not contain symptoms that are clearly inconsistent with the discharge diagnosis.",
                ),
                (
                    ("discharge_diagnosis", "hospital_course"),
                    "The discharge diagnosis should not conflict with examinations and tests in the hospital course; the hospital course should not contain findings that are clearly inconsistent with the discharge diagnosis.",
                ),
                (
                    ("discharge_diagnosis", "discharge_instructions"),
                    "The discharge diagnosis should not conflict with medications in the discharge instructions; the discharge instructions should not contain medications that are clearly inconsistent with the discharge diagnosis.",
                ),
            ],
        }

    def _parse_json(self, output):
        output = re.sub(r"<think>.*</think>", "", output, flags=re.DOTALL)
        m = re.search(r"```json([^`]*)```", output)
        if m:
            output = m.group(1).strip()
        else:
            m = re.search(r"{[^}]*}", output)
            if m:
                output = m.group(0)
            else:
                return None
        try:
            return json.loads(output)
        except JSONDecodeError:
            try:
                return json.loads(output.replace("'", '"'))
            except JSONDecodeError:
                return None

    def _generator(self, n):
        prompt = self.template["generator"].format(
            condition=self.icd2diag[self.icd],
        )
        emrs = []
        while len(emrs) < n:
            output = self._generate(prompt, dict(n=n - len(emrs), temperature=1))
            output = self._basic_check(output)
            emrs += output
        return list(enumerate(emrs))

    def _basic_check(self, emr):
        output = []
        fields = {
            "patient_info",
            "chief_complaint",
            "history_of_present_illness",
            "hospital_course",
            "discharge_diagnosis",
            "discharge_instructions",
        }
        for i in emr:
            i = self._parse_json(i)
            if isinstance(i, dict) and i:
                flag = 1
                for j in fields:
                    if j not in i or not isinstance(i[j], str):
                        flag = 0
                        break
                if flag:
                    output.append({j: i[j] for j in fields})
        return output

    def _corpus_critic(self, draft):
        draft = [i[1] for i in draft]
        count, inverse_map = self._get_corpus_statistic(draft)
        expect_count = {}
        for i in self.ref_dist[self.icd]:
            expect_count[i] = self.allocate_samples(
                self.ref_dist[self.icd][i],
                round(len(draft) * sum(self.ref_dist[self.icd][i].values())),
            )
        update_plan = [
            {
                "sex": {"add": None, "remove": None},
                "age": {"add": None, "remove": None},
            }
            | {i: {"add": set(), "remove": set()} for i in self.fields}
            for _ in draft
        ]
        if count["sex"]["male"] > expect_count["sex"]["male"]:
            for i in random.sample(
                inverse_map["sex"]["male"],
                count["sex"]["male"] - expect_count["sex"]["male"],
            ):
                update_plan[i]["sex"]["add"] = "female"
                update_plan[i]["sex"]["remove"] = "male"
            for i in inverse_map["sex"][None]:
                update_plan[i]["sex"]["add"] = "female"
        elif count["sex"]["female"] > expect_count["sex"]["female"]:
            for i in random.sample(
                inverse_map["sex"]["female"],
                count["sex"]["female"] - expect_count["sex"]["female"],
            ):
                update_plan[i]["sex"]["add"] = "male"
                update_plan[i]["sex"]["remove"] = "female"
            for i in inverse_map["sex"][None]:
                update_plan[i]["sex"]["add"] = "male"
        else:
            random.shuffle(inverse_map["sex"][None])
            for i in inverse_map["sex"][None][
                : expect_count["sex"]["male"] - count["sex"]["male"]
            ]:
                update_plan[i]["sex"]["add"] = "male"
            for i in inverse_map["sex"][None][
                expect_count["sex"]["male"] - count["sex"]["male"] :
            ]:
                update_plan[i]["sex"]["add"] = "female"

        candidate_age = []
        for i in expect_count["age"]:
            if expect_count["age"][i] > count["age"][i]:
                for j in range(expect_count["age"][i] - count["age"][i]):
                    candidate_age.append(random.randint(i * 5, i * 5 + 4))
        random.shuffle(candidate_age)
        for i in count["age"]:
            if i is None:
                for j in inverse_map["age"][i]:
                    update_plan[j]["age"]["add"] = candidate_age.pop()
            elif count["age"][i] > expect_count["age"][i]:
                for j in random.sample(
                    inverse_map["age"][i], count["age"][i] - expect_count["age"][i]
                ):
                    update_plan[j]["age"]["add"] = candidate_age.pop()
                    update_plan[j]["age"]["remove"] = i

        for field in self.fields:
            candidate = []
            for i in expect_count[field]:
                if expect_count[field][i] > count[field][i]:
                    candidate += [i] * (expect_count[field][i] - count[field][i])
            random.shuffle(candidate)
            for i in count[field]:
                if count[field][i] > expect_count[field][i] + 1:
                    for j in random.sample(
                        inverse_map[field][i], count[field][i] - expect_count[field][i]
                    ):
                        if candidate:
                            for k in candidate:
                                if (
                                    k not in draft[j][field]
                                    and k not in update_plan[j][field]["add"]
                                ):
                                    update_plan[j][field]["add"].add(k)
                                    candidate.remove(k)
                                    break
                        update_plan[j][field]["remove"].add(i)
            if candidate:
                index = list(range(len(draft)))
                random.shuffle(index)
                for i in index:
                    for j in candidate:
                        if (
                            j not in draft[i][field]
                            and j not in update_plan[i][field]["add"]
                        ):
                            update_plan[i][field]["add"].add(j)
                            candidate.remove(j)
                            break
                    if not candidate:
                        break

        feedback = []
        for i in update_plan:
            current_feedback = {}
            if i["sex"]["add"]:
                if i["sex"]["remove"]:
                    current_feedback.setdefault("patient_info", []).append(
                        "Please modify the gender to “{}”".format(i["sex"]["add"])
                    )
                else:
                    current_feedback.setdefault("patient_info", []).append(
                        "Please add the gender “{}”".format(i["sex"]["add"])
                    )
            if i["age"]["add"]:
                if i["age"]["remove"]:
                    current_feedback.setdefault("patient_info", []).append(
                        "Please modify the age to “{}”".format(i["age"]["add"])
                    )
                else:
                    current_feedback.setdefault("patient_info", []).append(
                        "Please add the age “{}”".format(i["age"]["add"])
                    )
            for field in self.fields:
                if i[field]["add"]:
                    current_feedback.setdefault(field, []).append(
                        "Add the following keywords：{keyword}".format(
                            keyword="、".join(i[field]["add"]),
                        )
                    )
                if i[field]["remove"]:
                    current_feedback.setdefault(field, []).append(
                        "Remove the following keywords: {keyword}".format(
                            keyword="、".join(i[field]["remove"]),
                        )
                    )
            feedback.append(current_feedback)
        return feedback

    def _split_seq(self, seq, mask):
        pos, neg = [], []
        for i, j in zip(seq, mask, strict=True):
            if j:
                pos.append(i)
            else:
                neg.append(i)
        return pos, neg

    def _load_ref(self, ref_path):
        self.ref_dist = defaultdict(
            lambda: (
                {
                    "sex": Counter(),
                    "age": Counter(),
                }
                | {i: Counter() for i in self.fields}
            )
        )
        with open(ref_path) as f:
            for line in f:
                line = json.loads(line)
                self.ref_dist[line["icd"]]["sex"][line["sex"]] += 1
                self.ref_dist[line["icd"]]["age"][int(line["age"]) // 5] += 1
                for i in self.fields:
                    self.ref_dist[line["icd"]][i].update(
                        set(sum(line["entity"][i].values(), []))
                    )
        for icd in self.ref_dist:
            total = sum(self.ref_dist[icd]["sex"].values())
            for i in self.ref_dist[icd]:
                for j in self.ref_dist[icd][i]:
                    self.ref_dist[icd][i][j] /= total

    def allocate_samples(self, prob_dist, n):
        allocation = Counter()

        residuals = {}
        current_total = 0

        for x_i, p_i in prob_dist.items():
            ideal_count = n * p_i
            allocation[x_i] = int(ideal_count)
            residuals[x_i] = ideal_count - allocation[x_i]
            current_total += allocation[x_i]

        remaining = n - current_total

        for _ in range(remaining):
            x_max_residual = max(residuals, key=residuals.get)
            allocation[x_max_residual] += 1
            residuals[x_max_residual] -= 1

        return allocation

    def _get_corpus_statistic(self, corpus):
        count = {i: Counter() for i in self.ref_dist[self.icd]}
        inverse_map = {
            "sex": defaultdict(list),
            "age": defaultdict(list),
        } | {i: defaultdict(list) for i in self.fields}
        for i, x in enumerate(corpus):
            feature = self._extract_feature(x)
            count["sex"][feature["sex"]] += 1
            count["age"][feature["age"]] += 1
            inverse_map["sex"][feature["sex"]].append(i)
            inverse_map["age"][feature["age"]].append(i)
            for j in self.fields:
                count[j].update(feature[j])
                for k in feature[j]:
                    inverse_map[j][k].append(i)
        return count, inverse_map

    def _extract_feature(self, emr):
        feature = {}
        feature["sex"] = None
        if "male" in emr["patient_info"].lower():
            feature["sex"] = "male"
        if "female" in emr["patient_info"].lower():
            feature["sex"] = "female"
        m = re.search(r"\d+", emr["patient_info"])
        if m:
            feature["age"] = int(m.group(0)) // 5
        else:
            feature["age"] = None
        for i in self.fields:
            feature[i] = []
            for j in self.ref_dist[self.icd][i]:
                if j in emr[i]:
                    feature[i].append(j)
        return feature

    def _section_critic(self, draft):
        requirement = self.criteria["completeness"]
        prompt = [
            self.template["critic"].format(
                field=i[0],
                requirement=i[1],
                emr={i[0]: j[1][i[0]]},
            )
            for j in draft
            for i in requirement
        ]
        feedback = self._generate(prompt)
        result = []
        for i in range(0, len(feedback), len(requirement)):
            temp = {}
            for j, k in zip(
                feedback[i : i + len(requirement)],
                requirement,
                strict=True,
            ):
                j = self._parse_json(j)
                if isinstance(j, dict):
                    meets = j.get("meets_requirement")
                    if meets is False:
                        temp.setdefault(k[0], []).append(k[1])
            result.append(temp)
        return result

    def _document_critic(self, draft):
        requirement = self.criteria["consistency"] + self.criteria["correctness"]
        prompt = [
            self.template["critic"].format(
                field="和".join(i[0]),
                requirement=i[1],
                emr={k: j[1][k] for k in i[0]},
            )
            for j in draft
            for i in requirement
        ]
        feedback = self._generate(prompt)
        result = []
        for i in range(0, len(feedback), len(requirement)):
            temp = {}
            for j, k in zip(
                feedback[i : i + len(requirement)],
                requirement,
                strict=True,
            ):
                j = self._parse_json(j)
                if isinstance(j, dict):
                    meets = j.get("meets_requirement")
                    if meets is False:
                        temp.setdefault(k[0], []).append(k[1])
            result.append(temp)
        return result

    def _adviser(self, draft, feedback):
        prompt = []
        length = []
        for i, j in zip(draft, feedback, strict=True):
            current_length = 0
            for field in j:
                for k in j[field]:
                    if isinstance(field, tuple):
                        prompt.append(
                            self.template["adviser"].format(
                                field=" and ".join(field),
                                emr={l: i[1][l] for l in field},
                                requirement=k,
                            )
                        )
                    else:
                        prompt.append(
                            self.template["adviser"].format(
                                field=field,
                                emr={field: i[1][field]},
                                requirement=k,
                            )
                        )
                    current_length += 1
            length.append(current_length)
        output = self._generate(prompt)
        result = []
        for i in length:
            current_feedback = {}
            for j in output[:i]:
                j = self._parse_json(j)
                if j:
                    if j["field_to_modify"] in {
                        "patient_info",
                        "chief_complaint",
                        "history_of_present_illness",
                        "hospital_course",
                        "discharge_diagnosis",
                        "discharge_instructions",
                    }:
                        current_feedback.setdefault(j["field_to_modify"], []).append(
                            j["modification_suggestion"]
                        )
            result.append(current_feedback)
            output = output[i:]
        return result

    def _reviser(self, emr, feedback):
        prompt = []
        for i, j in zip(emr, feedback, strict=True):
            for field in j:
                prompt.append(
                    self.template["reviser"].format(
                        field=field,
                        emr=i[1],
                        feedback=j[field],
                    )
                )

        output = self._generate(prompt)
        result = []
        for i, j in zip(emr, feedback, strict=True):
            for k, l in zip(output[: len(j)], j):
                k = self._parse_json(k)
                if k and l in k and isinstance(k[l], str):
                    i[1][l] = k[l]
            result.append(i)
            output = output[len(j) :]
        return result

    def generate_corpus(self, icd_count, max_revision, output_path):
        with open(output_path, "w") as f:
            for icd in icd_count:
                self.icd = icd
                draft = self._generator(icd_count[icd])
                for _ in range(max_revision):
                    feedback = self._corpus_critic(draft)
                    feedback = self._adviser(draft, feedback)
                    draft = self._reviser(draft, feedback)
                for _ in range(max_revision):
                    feedback = self._section_critic(draft)
                    feedback = self._adviser(draft, feedback)
                    draft = self._reviser(draft, feedback)
                for _ in range(max_revision):
                    feedback = self._document_critic(draft)
                    feedback = self._adviser(draft, feedback)
                    draft = self._reviser(draft, feedback)
                for i in draft:
                    json.dump(i[1], f, ensure_ascii=False)
                    f.write("\n")


if __name__ == "__main__":
    generator = LLMCARe(
        model_path="path/to/model",
        ref_path="example-ref.jsonl",
        mapping_path="example-map.json",
    )
    generator.generate_corpus(
        icd_count={"I63.900": 2, "I25.103": 3},
        max_revision=2,
        output_path="path/to/output",
    )
