# from pathlib import Path

# class FeedbackSystem:
#     def __init__(self, feedback_file: str = "./feedback/feedback.json"):
#         self.feedback_file = Path(feedback_file)
#         self.feedback_file.parent.mkdir(exist_ok=True)
#         self.feedback_data = self._load_feedback()

#         self.error_categories = {
#             "wrong_answer": "❌ Неправильный ответ",
#             "incomplete": "📝 Неполный ответ",
#             "irrelevant": "🔍 Не по теме",
#             "outdated": "📅 Устаревшая информация",
#             "format": "📄 Плохое форматирование",
#             "language": "🌍 Проблемы с языком",
#             "other": "🤔 Другое"
#         }