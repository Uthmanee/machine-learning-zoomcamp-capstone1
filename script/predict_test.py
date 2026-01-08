import requests

url = 'http://localhost:9696/predict'
customer = {'age': 18,
 'marital_status': 'single',
 'education_level': 'high_school',
 'number_of_children': 0,
 'smoking_status': 'non-smoker',
 'physical_activity_level': 'moderate',
 'employment_status': 'employed',
 'income': 30758.68,
 'alcohol_consumption': 'moderate',
 'dietary_habits': 'unhealthy',
 'sleep_patterns': 'fair',
 'history_of_mental_illness': 'no',
 'history_of_substance_abuse': 'no',
 'family_history_of_depression': 'no',
 'chronic_medical_conditions': 0}

response = requests.post(url, json=customer).json()
print(response)