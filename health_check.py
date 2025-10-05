"""
Скрипт для проверки работоспособности API после деплоя
"""
import requests
import sys

def check_api_health(base_url: str):
    """Проверяет работоспособность API"""
    try:
        # Проверяем основной endpoint
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API документация доступна")
        else:
            print(f"❌ API документация недоступна: {response.status_code}")
            return False
            
        # Проверяем health endpoint (если есть)
        try:
            health_response = requests.get(f"{base_url}/health")
            if health_response.status_code == 200:
                print("✅ Health check пройден")
            else:
                print(f"⚠️ Health check недоступен: {health_response.status_code}")
        except:
            print("⚠️ Health endpoint не найден (это нормально)")
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка подключения к API: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python health_check.py <API_URL>")
        print("Пример: python health_check.py https://your-app.onrender.com")
        sys.exit(1)
    
    api_url = sys.argv[1]
    print(f"Проверяем API: {api_url}")
    
    if check_api_health(api_url):
        print("🎉 API работает корректно!")
    else:
        print("💥 API не работает!")
        sys.exit(1)
