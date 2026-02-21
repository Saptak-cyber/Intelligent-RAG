# Test Questions for Model Router and Output Evaluator

Generated based on actual chunks in vector database (75 chunks from 30 ClearPath documents)

---

## Part 1: Model Router Test Questions

### Simple Questions (Should route to llama-3.1-8b-instant)

These questions should trigger the "default" rule - short, straightforward, single-topic queries:

1. What is ClearPath?
2. How do I create a new task?
3. What are the keyboard shortcuts?
4. How much does the Pro plan cost?
5. What is the PTO policy?
6. How do I reset my password?
7. What integrations are available?
8. How do I invite team members?
9. What is the mobile app called?
10. How do I export my data?
11. What is the cancellation policy?
12. How do I delete my account?
13. What are the support channels?
14. How do I change my email address?
15. What is the overtime policy?

### Complex Questions - Keyword Triggered (Should route to llama-3.3-70b-versatile)

Questions containing complex keywords: "explain", "compare", "analyze", "difference", "relationship"

16. Explain how custom workflows work in ClearPath.
17. Compare the Free and Pro plans.
18. Analyze the benefits of the Enterprise plan.
19. What's the difference between exempt and non-exempt employees?
20. Explain the relationship between webhooks and API integrations.
21. Compare the mobile app features on iOS vs Android.
22. Explain how the data security policy protects customer information.
23. What's the difference between monthly and annual billing?
24. Analyze the remote work eligibility requirements.
25. Compare the support response times across different plans.

### Complex Questions - Length Triggered (>15 words)

26. Can you tell me about the complete onboarding process for new team members including all the steps and timeline recommendations?
27. What are all the different features available in the Enterprise plan and how do they differ from the Pro plan features?
28. I need to understand the full data security policy including encryption requirements, access controls, and incident reporting procedures for my compliance team.
29. What is the complete process for setting up integrations with Slack, GitHub, and other tools, and what permissions are required?
30. Can you walk me through the entire billing cycle, payment methods, plan changes, and cancellation process with all the details?

### Complex Questions - Multiple Questions (>1 question mark)

31. How do I create a project? How do I add team members? How do I set up workflows?
32. What is the Pro plan price? What features does it include? Can I upgrade later?
33. How do I reset my password? What if I don't receive the email? Should I check spam?
34. What integrations are available? How do I connect them? Are they included in all plans?
35. How do I export data? What formats are supported? Will it include attachments?

### Complex Questions - Comparison Words

Questions containing: "versus", "vs", "better", "worse", "compared to"

36. Which is better for small teams, Free or Pro?
37. How does ClearPath compare to Asana?
38. Pro versus Enterprise - which should I choose?
39. Is annual billing better than monthly?
40. How does the mobile app compared to the web version?

### Out-of-Distribution (OOD) Questions (Should skip retrieval)

These should trigger the OOD filter and skip document retrieval:

41. Hi
42. Hello
43. Hey there
44. Thanks
45. Thank you
46. Who are you?
47. What can you do?
48. Help
49. Hello!
50. Thanks!

### Edge Cases

51. Hi, how do I reset my password? (Should NOT trigger OOD - legitimate question)
52. I need help with my server configuration. (Should NOT trigger OOD - specific request)
53. Can you help me understand the pricing? (Should NOT trigger OOD - "help" in context)
54. What is the CSV export format? (Should NOT trigger comparison - "csv" contains "vs")
55. The versus operator in SQL. (Should trigger comparison - "versus" is standalone)

---

## Part 2: Output Evaluator Test Questions

### Flag: `no_context`

**Condition:** chunks_retrieved == 0 AND response is not a refusal

Test by asking questions completely outside the document scope:

56. What is the weather in New York today?
57. Who won the Super Bowl in 2023?
58. What is the capital of France?
59. How do I bake a chocolate cake?
60. What is quantum computing?

**Expected:** If the LLM answers these without refusing, flag `no_context` should be raised.

---

### Flag: `refusal`

**Condition:** Response contains refusal phrases AND is not a partial answer

Refusal phrases: "i don't have", "not mentioned", "cannot find", "don't know", "no information", "i cannot", "i can't", "unable to find", "not available", "doesn't mention"

Test with questions that have no answer in the docs:

61. What is ClearPath's office address?
62. Who is the CEO of ClearPath?
63. What is the company's annual revenue?
64. How many customers does ClearPath have?
65. What is the employee count at ClearPath?

**Expected:** LLM should refuse to answer, triggering `refusal` flag.

**Counter-test (should NOT flag):** Partial answers with contrast words:

66. Does ClearPath support Jira integration? (Answer might be: "I don't have specific information about Jira, but the documentation mentions integrations with Slack, GitHub...")

---

### Flag: `unverified_feature`

**Condition:** Response mentions proper nouns (features/integrations) not found in retrieved chunks

Test by asking about features that might sound plausible but aren't in the docs:

67. Does ClearPath integrate with Microsoft Teams?
68. Can I use ClearPath with Notion?
69. Does ClearPath support Salesforce integration?
70. Is there a ClearPath plugin for VS Code?
71. Does ClearPath have a Chrome extension?
72. Can I integrate ClearPath with Zoom?
73. Does ClearPath support LDAP authentication?
74. Is there a ClearPath API for Python?
75. Does ClearPath integrate with Figma?
76. Can I use ClearPath with Azure DevOps?

**Expected:** If LLM mentions these integrations/features without them being in the retrieved chunks, flag `unverified_feature` should be raised.

**Verified features (should NOT flag):** Slack, GitHub, Jira, Trello, Asana, Monday, Google, Microsoft, Apple, Amazon

---

### Flag: `pricing_uncertainty`

**Condition:** Response is pricing-related AND (contains hedging language OR mentions conflicts)

Hedging phrases: "may", "might", "approximately", "around", "varies", "could be", "possibly", "perhaps", "roughly"

Conflict phrases: "conflict", "contradict", "different prices", "inconsistent", "discrepancy", "unclear", "not explicitly stated", "multiple prices listed"

Test with pricing questions:

77. How much does the Enterprise plan cost?
78. What is the exact price for 30 users?
79. Are there any hidden fees?
80. What is the cost for annual billing?
81. How much does extra storage cost?
82. What are the volume discount rates?
83. Is there a setup fee for Enterprise?
84. What is the price per additional user?

**Expected:** If the LLM uses hedging language like "approximately $2,500/month" or mentions "pricing varies" or "contact sales", the `pricing_uncertainty` flag should be raised.

**Clear pricing (should NOT flag):**

85. How much does the Pro plan cost? (Answer: $12/user/month - clear and specific)
86. What is the Free plan price? (Answer: $0 - no uncertainty)

---

## Part 3: Combined Router + Evaluator Tests

### Scenario 1: Simple + No Context
87. What is the population of Tokyo?
- **Router:** Simple (default rule)
- **Evaluator:** `no_context` (if LLM answers without docs)

### Scenario 2: Simple + Refusal
88. What is ClearPath's stock price?
- **Router:** Simple (default rule)
- **Evaluator:** `refusal` (no info in docs)

### Scenario 3: Complex + Unverified Feature
89. Compare ClearPath's integration with Salesforce versus HubSpot.
- **Router:** Complex (comparison words)
- **Evaluator:** `unverified_feature` (Salesforce/HubSpot not in docs)

### Scenario 4: Complex + Pricing Uncertainty
90. Explain the difference between Enterprise pricing for 50 users versus 100 users.
- **Router:** Complex (explain + difference keywords)
- **Evaluator:** `pricing_uncertainty` (Enterprise pricing is custom/varies)

### Scenario 5: OOD + No Flags
91. Hello
- **Router:** Simple + skip_retrieval (OOD filter)
- **Evaluator:** No flags (appropriate greeting response)

### Scenario 6: Simple + Verified Answer
92. What integrations does ClearPath support?
- **Router:** Simple (default rule)
- **Evaluator:** No flags (Slack, GitHub, etc. are in docs)

### Scenario 7: Complex + Verified Answer
93. Compare the storage limits between Free and Pro plans.
- **Router:** Complex (comparison words)
- **Evaluator:** No flags (500MB vs 50GB is in docs)

### Scenario 8: Length + Pricing Uncertainty
94. Can you provide a detailed breakdown of all the costs associated with the Enterprise plan including setup fees, per-user pricing, and any additional charges?
- **Router:** Complex (>15 words)
- **Evaluator:** `pricing_uncertainty` (Enterprise pricing is custom)

### Scenario 9: Multiple Questions + Partial Refusal
95. What is the CEO's name? What is the company address? What integrations are available?
- **Router:** Complex (multiple question marks)
- **Evaluator:** Likely partial answer (can answer integrations but not CEO/address)

### Scenario 10: Simple + All Clear
96. How do I create a new task?
- **Router:** Simple (default rule)
- **Evaluator:** No flags (clear answer in docs)

---

## Testing Instructions

### For Model Router Testing:
1. Send each question to the `/query` endpoint
2. Check the `debug_info.routing` field in the response
3. Verify `category` matches expected (simple/complex)
4. Verify `rule_triggered` matches expected rule
5. Verify `skip_retrieval` is true only for OOD questions

### For Output Evaluator Testing:
1. Send each question to the `/query` endpoint
2. Check the `debug_info.evaluation_flags` field in the response
3. Verify expected flags are present
4. Verify no unexpected flags are raised
5. Check `debug_info.chunks_retrieved` to understand context

### Success Criteria:
- **Router Accuracy:** >90% of questions route to expected model
- **OOD Detection:** 100% of greetings/meta-comments skip retrieval
- **Evaluator Precision:** Flags raised only when appropriate
- **Evaluator Recall:** All quality issues are caught

---

## Document Coverage Summary

Your vector database contains information about:

**Employee/HR Docs (6):**
- Employee Handbook, Data Security Policy, Remote Work Guidelines, Code of Conduct, PTO Policy

**Product Documentation (16):**
- User Guide, Getting Started, Advanced Features, Integrations, Mobile App, Keyboard Shortcuts, Custom Workflows, Reporting & Analytics

**Pricing/Sales (4):**
- Pricing Sheet 2024, Enterprise Plan Details, Feature Comparison Matrix, FAQ

**Support (4):**
- Support SLA, Troubleshooting Guide, Account Management FAQ, Onboarding Checklist

**Internal/Engineering (6):**
- Q4 2023 Retrospective, Engineering Team Structure, Weekly Standup Notes, Product Roadmap 2024, API Documentation, Webhook Integration Guide, System Architecture, Deployment Infrastructure, Release Notes

**Known Integrations in Docs:**
Slack, GitHub, Jira, Trello, Asana, Monday, Notion, Confluence, Google, Microsoft, Apple, Amazon, Salesforce (mentioned in integration patterns)

**Known Pricing:**
- Free: $0, up to 5 users, 500MB storage
- Pro: $12/user/month, up to 25 users, 50GB storage
- Enterprise: Custom pricing, starts at $2,500/month for 50-100 users, 12-month minimum

Use this information to craft additional test questions as needed!
