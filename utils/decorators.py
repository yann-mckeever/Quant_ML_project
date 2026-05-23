import time
import logging
from functools import wraps
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)

def time_it(func):
    """ 
     Décorateur qui chronomètre le temps d'éxécution d'une fonction
     Nécéssaire pour surveiller le temsp d'entraînement d'un modèle

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time=time.time()

        result = func(*args,**kwargs)

        end_time=time.time()

        logging.info(f"[TIMER] La fonction '{func.__name__}' a mis {end_time - start_time:.4f} secondes")

        return result

    return wrapper 


def log_execution(func):
    """ 
    Décorateur qui trace l'éxécution, utile pour savoir où en est le programme
    """
    @wraps(func)
    def wrapper(*args,**kwargs):
        logging.info(f"[EXEC] Démarrage de '{func.__name__}' ...")
        result=func(*args,**kwargs)
        logging.info(f"[EXEC] Fin de '{func.__name__}'")
        return result
    return wrapper

def require_dataframe(func):
    """ 
    Décorateur de sécurité (Type checking dynamique)
    Vérifie que la fonction renvoie bien un dataframe non vide.
    """

    @wraps(func)

    def wrapper(*args,**kwargs):

        result=func(*args,**kwargs)

        if not isinstance(result, pd.DataFrame):
            logging.error(f"[DATA ERROR] '{func.__name__}' devrait renvoyer un DataFrame mais à renvoyer {type(result)}")
            raise TypeError("Le retour n'est pas un DataFrame Pandas")
        
        if result.empty :
            logging.warning(f"[DATA WARNING] Le DataFrame retourné par '{func.__name__}' est vide ")

        return result
    return wrapper




