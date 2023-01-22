#bin/bash

echo Probando con las imagenes de la clase Glandular-denso
python -m src.mamSegClas test_images/Glandular-denso/*.jpg

echo Probando con las imagenes de la clase Glandular-graso
python -m src.mamSegClas test_images/Glandular-graso/*.jpg

echo Probando con las imagenes de la clase Graso
python -m src.mamSegClas test_images/Graso/*.jpg
