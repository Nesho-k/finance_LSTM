# Configuration CORS pour le Frontend

## Vue d'ensemble

Le backend accepte maintenant les requêtes depuis des origines spécifiques via la variable d'environnement `ALLOWED_ORIGINS`.

## Configuration

### 1. En développement local

Créez un fichier `.env` à la racine du projet :

```bash
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:5174
```

### 2. Sur Render (Production)

#### Option A : Via le Dashboard Render (Recommandé)

1. Allez sur votre service sur https://dashboard.render.com
2. Cliquez sur **"Environment"** dans le menu de gauche
3. Cliquez sur **"Add Environment Variable"**
4. Ajoutez :
   - **Key** : `ALLOWED_ORIGINS`
   - **Value** : `https://votre-app.vercel.app,https://votre-app-git-main.vercel.app,https://votre-app-previews.vercel.app`

5. Cliquez sur **"Save Changes"**
6. Le service redémarrera automatiquement

#### Option B : Via render.yaml

Le fichier `render.yaml` contient déjà la configuration. Modifiez la ligne 15 :

```yaml
- key: ALLOWED_ORIGINS
  value: https://votre-app.vercel.app,http://localhost:3000
```

Remplacez `https://votre-app.vercel.app` par votre vraie URL Vercel.

## URLs Vercel à autoriser

Vercel crée plusieurs URLs pour chaque projet :

1. **URL de production** : `https://votre-app.vercel.app`
2. **URL de la branche main** : `https://votre-app-git-main-username.vercel.app`
3. **URLs de preview** (branches) : `https://votre-app-git-branch-username.vercel.app`

### Exemple complet

Pour accepter toutes les URLs Vercel + localhost :

```
ALLOWED_ORIGINS=https://meteo-app.vercel.app,https://meteo-app-git-main.vercel.app,https://meteo-app-*.vercel.app,http://localhost:3000,http://localhost:5173
```

⚠️ **Note** : Les wildcards (`*`) ne sont pas supportés par CORS. Vous devez lister chaque URL explicitement.

### Solution alternative : Accepter toutes les URLs Vercel

Si vous avez beaucoup de preview deployments, vous pouvez modifier le code pour accepter dynamiquement toutes les URLs `.vercel.app` :

#### Modifier `src/api.py`

```python
import re

# Configuration CORS dynamique
def is_allowed_origin(origin: str) -> bool:
    """Vérifie si l'origine est autorisée"""
    allowed_patterns = [
        r"^http://localhost:\d+$",  # Localhost
        r"^https://.*\.vercel\.app$",  # Toutes les URLs Vercel
    ]
    return any(re.match(pattern, origin) for pattern in allowed_patterns)

# CORS avec validation dynamique
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(http://localhost:\d+|https://.*\.vercel\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Trouver votre URL Vercel

1. Déployez votre frontend sur Vercel
2. Allez sur https://vercel.com/dashboard
3. Sélectionnez votre projet
4. L'URL principale est affichée en haut : `https://votre-app.vercel.app`

## Tester le CORS

### Depuis la console du navigateur

```javascript
fetch('https://votre-backend.onrender.com/health')
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error('CORS error:', err));
```

Si vous voyez une erreur CORS, vérifiez que :
1. L'URL du frontend est bien dans `ALLOWED_ORIGINS`
2. Le backend a bien redémarré après la modification
3. L'URL correspond exactement (avec/sans trailing slash, http vs https)

### Vérifier les headers

Dans les DevTools du navigateur :
1. Ouvrez **Network** (Réseau)
2. Faites une requête vers votre API
3. Regardez les headers de la réponse :
   - `Access-Control-Allow-Origin` doit contenir votre URL frontend
   - `Access-Control-Allow-Credentials` doit être `true`

## Dépannage

### Erreur : "CORS policy: No 'Access-Control-Allow-Origin' header"

**Cause** : L'URL du frontend n'est pas dans `ALLOWED_ORIGINS`

**Solution** :
1. Vérifiez l'URL exacte du frontend dans la console du navigateur
2. Ajoutez cette URL à `ALLOWED_ORIGINS` sur Render
3. Redémarrez le service

### Erreur : "CORS policy: Credentials flag is 'true'"

**Cause** : `allow_credentials=True` avec `allow_origins=["*"]`

**Solution** : C'est déjà corrigé ! Le code utilise maintenant des origines spécifiques.

### Frontend en développement fonctionne mais pas en production

**Cause** : L'URL de production Vercel n'est pas dans `ALLOWED_ORIGINS`

**Solution** : Ajoutez l'URL de production Vercel à la variable d'environnement sur Render.

## Sécurité

### ✅ Bonnes pratiques (implémentées)

- Origines spécifiques au lieu de `*`
- Variable d'environnement pour faciliter les changements
- Localhost autorisé uniquement en dev

### ⚠️ À éviter

- `allow_origins=["*"]` en production (vulnérable aux attaques CSRF)
- Autoriser des domaines non-HTTPS en production

## Exemple de configuration complète

### Fichier `.env` (dev local)

```bash
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Render Environment Variables (production)

```
ALLOWED_ORIGINS=https://meteo-app.vercel.app,https://www.meteo-app.com
```

## Logs de debug

Pour vérifier quelle origine est autorisée, vous pouvez temporairement ajouter un log dans `api.py` :

```python
@app.middleware("http")
async def log_origin(request, call_next):
    origin = request.headers.get("origin")
    print(f"Request from origin: {origin}")
    response = await call_next(request)
    return response
```

Consultez ensuite les logs sur Render : **Dashboard → Logs**

## Ressources

- [FastAPI CORS Docs](https://fastapi.tiangolo.com/tutorial/cors/)
- [MDN CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [Render Environment Variables](https://render.com/docs/environment-variables)
