import unittest

from agent import agent


class EmailAgentTests(unittest.TestCase):
    def test_ignore_marketing_email(self):
        result = agent({"email_text": "Exclusive offer! Buy now and save 70% on our premium plan. Unsubscribe anytime."})
        self.assertEqual(result["classification"], "ignore")
        self.assertEqual(result["status"], "success")
        self.assertIsInstance(result["message"], str)
        self.assertIsNone(result["draft_response"])

    def test_notify_financial_email(self):
        result = agent({"email_text": "Urgent: payment overdue. Your invoice is past due and legal action may follow."})
        self.assertEqual(result["classification"], "notify")
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["draft_response"])

    def test_respond_customer_email(self):
        result = agent({"email_text": "Hi, could we schedule a meeting next Tuesday to discuss the new proposal?"})
        self.assertEqual(result["classification"], "respond")
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["draft_response"])
        self.assertIn("Thank you", result["draft_response"])

    def test_failure_on_blank_input(self):
        result = agent({"email_text": ""})
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["draft_response"])


if __name__ == "__main__":
    unittest.main()
