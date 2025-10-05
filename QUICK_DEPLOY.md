# 🚀 Быстрый деплой на Render

## Пошаговая инструкция:

### 1. Подготовка (5 минут)
```bash
# Убедитесь, что все файлы закоммичены
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Создание сервиса на Render (10 минут)

1. **Зайдите на [render.com](https://render.com)**
2. **Нажмите "New +" → "Web Service"**
3. **Подключите GitHub репозиторий**
4. **Настройте параметры:**

```
Name: explore-planets-api
Environment: Python 3
Build Command: pip install -r requirements.txt && alembic upgrade head
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Создание базы данных (5 минут)

1. **В панели Render → "New +" → "PostgreSQL"**
2. **Настройки:**
   ```
   Name: explore-planets-db
   Plan: Free
   Database Name: explore_planets
   User: explore_planets_user
   ```

### 4. Настройка переменных окружения (2 минуты)

В настройках веб-сервиса добавьте:
```
DATABASE_URL = <скопируйте из настроек базы данных>
PYTHON_VERSION = 3.11.0
```

### 5. Деплой (5 минут)

1. **Нажмите "Deploy"**
2. **Дождитесь завершения** (обычно 5-10 минут)
3. **Проверьте логи** на наличие ошибок

### 6. Проверка работы

После деплоя проверьте:
- **API:** `https://your-app-name.onrender.com/health`
- **Документация:** `https://your-app-name.onrender.com/docs`

## 🔧 Возможные проблемы:

### База данных не подключается
- Проверьте, что `DATABASE_URL` правильно скопирован
- Убедитесь, что база данных запущена

### Ошибки миграций
- Проверьте логи сборки
- Убедитесь, что `alembic upgrade head` выполнился успешно

### Модели ML не загружаются
- Убедитесь, что файлы моделей в репозитории
- Проверьте пути в `config.py`

## 📊 Мониторинг:

- **Логи:** Render Dashboard → Your Service → Logs
- **Метрики:** Render Dashboard → Your Service → Metrics
- **База данных:** Render Dashboard → Your Database → Connect

## 🎯 Готово!

Ваш API будет доступен по адресу: `https://your-app-name.onrender.com`
