from tqdm import tqdm
import re
from difflib import SequenceMatcher
from collections import defaultdict
import re
from dateutil.parser import parse


class TextCapsBleu4Evaluator:
    def __init__(self):
        # The following script requires Java 1.8.0 and pycocotools installed.
        # The pycocoevalcap can be installed with pip as
        # pip install git+https://github.com/ronghanghu/coco-caption.git@python23
        # Original pycocoevalcap code is at https://github.com/tylin/coco-caption
        # but has no python3 support yet.
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        except ModuleNotFoundError:
            print(
                "Please install pycocoevalcap module using "
                "pip install git+https://github.com/ronghanghu/coco-caption.git@python23"  # noqa
            )
            raise

        self.tokenizer = PTBTokenizer()
        # self.scorer = Bleu(4)
        self.scorer = Bleu(1)

    def set_q(self, Q):
        self.Q = Q

    def eval_pred_list(self, pred_list):
        # Create reference and hypotheses captions.
        gts = {}
        res = {}
        for idx, entry in enumerate(pred_list):
            if (entry["question"] == self.Q):
                gts[idx] = [{"caption": a} for a in entry["gt_answers"]]
                res[idx] = [{"caption": entry["pred_answer"]}]
        
        # you need to install Java
        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)
        score, _ = self.scorer.compute_score(gts, res)

        
        bleu1 = score[0]  # score is (Bleu-1, Bleu-2, Bleu-3, Bleu-4)
        # print("score:", score)
        # print("bleu:", bleu1)
        return bleu1


class STVQAANLSEvaluator:
    def __init__(self):
        import editdistance  # install with `pip install editdistance`

        self.get_edit_distance = editdistance.eval

    def set_q(self, Q):
        self.Q = Q
        
    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= 0.5 else 0.0
        return anls

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            if (entry["question"] == self.Q):
                anls = max(
                    self.get_anls(entry["pred_answer"], gt) for gt in entry["gt_answers"]
                )
                pred_scores.append(anls)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class TextMatchEvaluator:
    """Evaluates answer accuracy with text normalization and flexible matching.
    
    Features:
    - Case normalization
    - Punctuation removal
    - Whitespace cleaning
    - Substring matching
    - Partial similarity threshold
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Args:
            similarity_threshold: Minimum similarity score (0-1) for partial matches
        """
        self.similarity_threshold = similarity_threshold
        """
        Args:
            question_ids: List of question identifiers (Q1-Q8)
        """
        # self.question_ids = [f"Q{i}" for i in range(1, 9)]
        # print(self.question_ids)
        self.correct_counts = defaultdict(int)
        self.total_counts = defaultdict(int)


    def normalize_text(self, text: str) -> str:
        """Standardizes text for comparison"""
        # Case folding and whitespace normalization
        text = text.lower().strip()
        # Remove all punctuation except word boundaries
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse multiple spaces
        return re.sub(r'\s+', ' ', text)

    def text_similarity(self, a: str, b: str) -> float:
        """Calculates similarity ratio between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    def is_correct(self, gt: str, pred: str) -> bool:
        """Determines if prediction matches ground truth with flexible rules"""
        norm_gt = self.normalize_text(gt)
        norm_pred = self.normalize_text(pred)
        
        # Direct match after normalization
        if norm_gt == norm_pred:
            return True
            
        # Substring containment
        if norm_gt in norm_pred or norm_pred in norm_gt:
            return True
            
        # Partial similarity match
        if self.text_similarity(norm_gt, norm_pred) >= self.similarity_threshold:
            return True
            
        return False

    def extract_after_is(self, text: str, keyword: str = "is") -> str:
        """
        Extracts text after specified keyword with normalization.
        
        Args:
            text: Input string containing the answer
            keyword: Target keyword to search for (default: "is")
            
        Returns:
            Extracted text after keyword, or empty string if not found
        
        Examples:
            >>> extract_after_is("The vase is ATHENIAN.")
            'athenian'
        """
        # Split text into parts after first occurrence of keyword [8,9](@ref)
        parts = text.lower().partition(keyword.lower())
        
        if not parts[1]:  # Keyword not found
            return ""
        
        # Clean and normalize the result [5,6](@ref)
        extracted = parts[2].strip(' .,;:\t\n')  # Remove surrounding whitespace/punctuation
        return extracted.split(maxsplit=1)[0] if extracted else ""


    def _process_gt_answer(self, gt_text: str) -> str:
        """Extracts key information from ground truth answer"""
        return self.extract_after_is(gt_text.lower())

    def _validate_entry(self, entry: dict) -> bool:
        """Validates entry structure"""
        required_keys = ['question', 'gt_answers', 'pred_answer']
        return all(key in entry for key in required_keys)

    @staticmethod
    def extract_after_is(text: str) -> str:
        """Extracts text after 'is' with normalization"""
        parts = text.lower().partition('is')
        return parts[2].strip(' .;:,').split()[0] if parts[2] else ''

    @staticmethod
    def is_correct(gt: str, pred: str) -> bool:
        """Flexible text matching with normalization"""
        return gt in pred or pred in gt

    def _compute_final_accuracy(self, question_ids) -> dict:
        """Computes accuracy percentages with division guard"""
        accuracy = {}
        accuracy_list = []
        # count = 1
        for q in question_ids:
            total = self.total_counts.get(q, 0)
            correct = self.correct_counts.get(q, 0)
            acc = correct / total if total > 0 else 0.0
            accuracy[q] = acc
            accuracy_list.append(acc)
        return accuracy, accuracy_list
    
    def eval_pred_list(self, pred_list: list[str]) -> float:
        """Main method to calculate per-question accuracy
        
        Args:
            pred_list: List of prediction entries with:
                - question: Question identifier (Q1-Q8)
                - gt_answers: Ground truth text
                - pred_answer: Model prediction text
                
        Returns:
            Dictionary of accuracies for each question
        """

        Q1 = "What is the fabric of the vase?"
        Q2 = "What is the technique of the vase?"
        Q3 = "What is the shape name of the vase?"
        Q4 = "What is the provenance of the vase?"
        Q5 = "What is the date of the vase?"
        Q6 = "What is the attributed to of the vase?"
        Q7 = "What is the decoration of the vase?"
        # Q8 = "What is the overall of the vase?"

        question_ids = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
        # Initialize counters
        self.correct_counts.clear()
        self.total_counts.clear()
        
        
        for entry in tqdm(pred_list):
            if not self._validate_entry(entry):
                continue

            q_id = entry['question']
            if q_id not in question_ids:
                continue
            
            # Update total count for this question
            self.total_counts[q_id] += 1

            # Handle empty predictions
            if not entry['pred_answer'].strip():
                continue

            # Check for special "not available" case
            if "not available" in entry['gt_answers'].lower():
                self.correct_counts[q_id] += 1
                continue

            if (entry["question"] == Q5) or (entry["question"] == Q7):
                continue
            
            gt_answers = self.extract_after_is(entry["gt_answers"])
            if self.is_correct(gt_answers, entry["pred_answer"]):
                # correct += 1
                self.correct_counts[q_id] += 1
            

        # Calculate final accuracy scores
        return self._compute_final_accuracy(question_ids)
        # return correct / len(pred_list)
    


class DateAccuracyEvaluator:
    def __init__(self):
        self.date_pattern = r"(\b\d{3,4}\s*(BCE?|AD)?\b)(\s*to\s*|\s*-\s*)(\b\d{3,4}\s*(BCE?|AD)?\b)"
    
    def _convert_number(self, value: str) -> int:
        """
        Convert a numeric string with optional era notation (BC/BCE/AD/CE) to integer.
        
        Handles:
        - BC/BCE as negative years
        - AD/CE as positive years (default if no era specified)
        - Pure numeric strings
        - Thousand separators (commas)
        
        Args:
            value: Input string containing numbers and optional era notation
            
        Returns:
            Integer representation of the input value
            
        Raises:
            ValueError: If no numeric value can be extracted
        """
        # Normalize era notations and remove commas
        cleaned = value.upper().replace(',', '').strip()
        
        # Identify era markers
        era_multiplier = 1
        for era in ['BC', 'BCE']:
            if era in cleaned:
                cleaned = cleaned.replace(era, '')
                era_multiplier = -1
                break
                
        # Extract numeric part using regular expression
        match = re.search(r'-?\d+', cleaned)
        if not match:
            raise ValueError(f"No numeric value found in: {value}")
            
        return int(match.group()) * era_multiplier
    
    def _extract_date_range(self, text):
        """使用正则表达式提取日期范围"""
        matches = re.findall(self.date_pattern, text, re.IGNORECASE)
        if not matches:
            return None
            
        try:
            # 转换纪元标识为负数
            def convert_year(y):
                if 'BC' in y.upper():
                    return -int(y.replace('BC', '').strip())
                return int(y.replace('AD', '').strip())
                
            start = convert_year(matches[0][0])
            end = self.convert_number(matches[0][3])
            return sorted([start, end])
        except:
            return None

    def _compare_ranges(self, gt_range, pred_range):
        """验证日期范围重叠度"""
        if not pred_range:
            return 0.0
            
        # 计算重叠比例
        overlap_start = max(gt_range[0], pred_range[0])
        overlap_end = min(gt_range[1], pred_range[1])
        overlap = max(0, overlap_end - overlap_start)
        total_span = gt_range[1] - gt_range[0]
        return overlap / total_span if total_span !=0 else 0.0

    def set_q(self, Q):
        self.Q = Q

    def eval_pred_list(self, pred_list):
        """主评估方法"""
        scores = []
        for entry in pred_list:
            if (entry["question"] == self.Q):
                # 提取真实日期范围
                gt_dates = [self._extract_date_range(gt) for gt in entry["gt_answers"]]
                gt_dates = [d for d in gt_dates if d]
                
                if not gt_dates:
                    scores.append(1.0 if "not available" in entry["gt_answers"] else 0.0)
                    continue
                    
                # 提取预测日期
                pred_range = self._extract_date_range(entry["pred_answer"])
                if not pred_range:
                    scores.append(0.0)
                    continue
                    
                # 计算最高匹配度
                max_score = max(self._compare_ranges(gt, pred_range) for gt in gt_dates)
                scores.append(1.0 if max_score >= 0.5 else 0.0)  
        return sum(scores) / len(scores)
