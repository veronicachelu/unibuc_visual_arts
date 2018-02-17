# Invatare automata in arta vizuala

## Setup minimal
### Install Python 3.6: 
* Pentru instalarea Python 3.6 puteti folosi direct Anaconda (https://www.anaconda.com/download)

### Virtual environment:
* Recomandam folosirea de virtual environments pentru a avea un setup clean intre mai multe proiecte.
* Pentru informatii legate de setarea unui nou virtual environment si vizualizarea environmenturilor curente aici. 
* TL;dr

        conda create -n py36 python=3.6 anaconda # creare nou virt env py36
        source activate py36 			   # activare virt env py36
        python --version  		           # verificare vers python
        pip install -r requirements.txt          # Install dependendinte
     
### Jupyter Notebook 
 
* Un ```Notebook``` reprezinta un document care contine cod si alte elemente precum figuri, linkuri, ecuatii.
* Aceste documente sunt produse de Jupyter Notebook App.
 
* Jupyter suporta mai multe limbaje de programare, printre care si Python, R, Julia etc.
 
* Componentele principale ale unui environment sunt notebook-urile si aplicatia.
* In acelasi timp mai exista si un kernel si un dashboard.
 
#### Jupyter Notebook App
* Este o aplicatie client-server ce permite editarea si rularea notebook-urilor in browser.
* Poate fi executata pe PC fara acces la internet sau poate fi instalata pe un server remote, accesat online.
* Componentele sale principale sunt kernelele si dashboard-ul
* Un kernel este un program care ruleaza si interpreteaza codul utilizatorului. 
* Exista kernele pentru fiecare virtual environment creat cu conda.
* Dashboard-ul iti permite sa vizualizezi notebook-urile pe care le-ai creat si pe care le poti redeschide. 
* De asemenea iti permite management-ul kernelelor.
 
* Jupyter vine deja instalat cu distributia Anaconda
* Pentru rulare: 
	
	    Jupyter notebook
 
#### Anaconda
 
* Anaconda permite accesul la peste 720 de pachete care pot fi instalate foarte usor cu conda, un manager de pachete, dependinte si environment-uri virtuale pentru Windows, Linux, MacOS. 
 
 
#### PyCharm 
* PyCharm este un IDE pentru python. Recomandam in special pentru debugger si insistam in a nu face debug cu printf. Pentru toate feature-urile disponibile aici exista o lista exhaustiva.

		


