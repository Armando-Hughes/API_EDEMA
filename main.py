import os
import tkinter as tk
import keras.models
from ttkwidgets.autocomplete import AutocompleteCombobox
from scipy import signal
import matplotlib.pyplot as plt
import pywt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from os import remove, rmdir

model = "img/modelos/modelo_densenet_LSTM.h5"
modelo= keras.models.load_model(model)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def modelo_fondo(nombre_img):
    img = image.load_img(nombre_img, target_size=(224, 224))
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    weights_fondo ="img/modelos/pesos_fondo.h5"
    modelo.load_weights(weights_fondo)
    y = modelo.predict(x)
    return y
def modelo_erg(nombre_img):
    img = image.load_img(nombre_img, target_size=(224, 224))
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    weights_fondo ="img/modelos/pesos_ERG.h5"
    modelo.load_weights(weights_fondo)
    y = modelo.predict(x)
    return y

def escalogramas(nombre, aparato_erg,llegada, salida):
    if aparato_erg == 'RETeval (LKC Technologies)':
        with open(salida+nombre, encoding="utf8", errors='ignore') as f:
            contents = f.read()
            l = []
            s = contents.split('\n')
            s1 = s[29:-1]
            for i in s1:
                lista1 = i.split(',')
                voltaje = lista1[3]
                l.append(float(voltaje))
            l1 = l[0:len(l)]
            x = l1[round(len(l1) * 0.25):round(len(l1) * 0.5)]
    else:
        with open(salida+nombre, encoding="utf8", errors='ignore') as f:
            contents = f.read()
            l = []
            s = contents.split('\n')
            s1 = s[3:-1]
            for i in s1:
                s2 = i.split('\t')
                voltaje = s2[1]
                l.append(float(voltaje))
            l1 = l[3:-1]
            x = l1[60001:len(l)]
    fs = 1000  # Hz
    T = 1/fs   # s
    Wn = [0.3, 40]  # Hz
    b, a = signal.butter(4, Wn, 'bandpass', fs=fs)
    y = signal.filtfilt(b, a, x)
    wavelet = 'morl'
    scales = np.arange(1, 256)
    filtered_coeffs, filtered_freqs = pywt.cwt(y, scales, wavelet, T)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.abs(filtered_coeffs), extent=[0, T, 0, np.max(scales)], cmap='coolwarm', aspect='auto', vmax=np.abs(filtered_coeffs).max())
    nombre1 = nombre.split('.')
    fig = plt.gcf()
    fig.savefig(llegada+nombre1[0], dpi='figure')
    plt.close()

def predicciones(nombre, aparato_erg):
    ruta = 'lista_pacientes'
    dir = './val/'+nombre+'/'
    dir2 = './val/'+ nombre + '/erg/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    lista = os.listdir(ruta)
    if nombre in lista:
        lista1 = os.listdir('lista_pacientes/'+nombre+'/')
        for i in lista1:
            i1 = i.split('.')
            if i1[1] == ('jpg' or 'png'):
                im = Image.open(ruta + '/' + nombre + '/' + i)
                alto = im.size[0]
                ancho = im.size[1]
                dif = abs(alto - ancho)
                img_max = max(alto, ancho)
                tupla1 = (round(dif / 2), 0, (ancho - round(dif / 2)), alto)
                tupla2 = (0, round(dif / 2), ancho, (alto - round(dif / 2)))
                if img_max == ancho:
                    im_recortada = im.crop(tupla1)
                    im_recortada.save(dir + i)
                else:
                    im_recortada = im.crop(tupla2)
                    im_recortada.save(dir + i)

            elif i1[1] == 'csv':
                    escalogramas(i, aparato_erg, dir2, ruta+'/'+nombre+'/')
    for i2 in os.listdir(dir2):
        img = Image.open(dir2+i2)
        img_recortada=img.crop((130, 70, 890, 445))
        img_recortada.save(dir+i2)
    for k in os.listdir(dir2):
        remove(dir2 + k)
    rmdir(dir2)
    ruta_fin = dir
    lista_fin = os.listdir(dir)
    for i3 in lista_fin:
        i3_1 = i3.split('.')
        i3_2 = i3_1[0].split('_')
        if (i3_2[0] == 'ERG' and i3_2[1] == 'OD'):
            erg_od = 0.3*modelo_erg(ruta_fin + i3)
        elif (i3_2[0] == 'ERG' and i3_2[1] == 'OI'):
            erg_oi = 0.3*modelo_erg(ruta_fin + i3)
        elif (i3_2[0] == 'FONDO' and i3_2[1] == 'OD'):
            fondo_od = 0.7*modelo_fondo(ruta_fin + i3)
        elif (i3_2[0] == 'FONDO' and i3_2[1] == 'OI'):
            fondo_oi = 0.7*modelo_fondo(ruta_fin + i3)
    try:
        y_final_od = erg_od + fondo_od
        y_final1_od = np.argmax(y_final_od)
    except:
        y_final1_od = 'No hay suficientes datos\n del Ojo Derecho'
    try:
        y_final_oi = erg_oi + fondo_oi
        y_final1_oi = np.argmax(y_final_oi)
    except:
        y_final1_oi ='No hay suficientes datos\n del Ojo Izquierdo'

    w = [y_final1_od, y_final1_oi]

    for k1 in os.listdir('val/'+ nombre+'/'):
        remove(dir+k1)
    rmdir(dir)

    return w
def inicio ():
    ventana_inicio = tk.Tk()
    ventana_inicio.geometry("760x600+0+0")
    ventana_inicio.resizable(width=False, height=False)
    ventana_inicio.title('PREDICTOR DE EMD')
    fondo1 = tk.PhotoImage(file="img/fondo.gif")
    fondo1_1 = tk.Label(ventana_inicio, image=fondo1).place(relx=0, rely=0, relwidth=1, relheight=1)
    label1 = tk.Label(ventana_inicio, text='PREDICTOR DE EDEMA MACULAR DIABÉTICO USANDO \nINTELIGENCIA ARTIFICIAL', font= ('Arial Black', 17))
    label1.place(relx=0.03, rely=0.1)
    label2 = tk.Label(ventana_inicio, text='El modelo utiliza Redes Neuronales Convolucionales Híbridas', font= ('Arial', 16))
    label2.place(relx=0.12, rely=0.3)
    label3 = tk.Label(ventana_inicio, text='Se requiere imagen de fondo de ojo en formato .jpg o .png \ny electrorretinograma basal en formato .csv',
                      font=('Arial', 14))
    label3.place(relx=0.16, rely=0.45)
    def ventana2():
        ventana1 = tk.Frame(ventana_inicio)
        ventana1.pack(expand=True, fill='both')
        fondo1 = tk.PhotoImage(file="img/fondo1.gif")
        fondo1_1 = tk.Label(ventana1, image=fondo1).place(relx=0, rely=0, relwidth=1, relheight=1)

        label1 = tk.Label(ventana1, text='SELECCIONE EL NOMBRE DEL PACIENTE',
                          font=('Arial Black', 17))
        label1.place(relx=0.15, rely=0.1)

        nombre = tk.StringVar()
        nombre1 = AutocompleteCombobox(ventana1, textvar=nombre, state='readonly',
                                        font=("Arial", 14), width=40)
        nombre1["completevalues"] = os.listdir('lista_pacientes')
        nombre1.place(relx=0.2, rely=0.2)

        label2 = tk.Label(ventana1, text='SELECCIONE EL ELECTRORRETINÓGRAFO UTILIZADO',
                          font=('Arial Black', 13))
        label2.place(relx=0.16, rely=0.35)

        label3 = tk.Label(ventana1, text='El proceso puede tardar unos minutos...\n No cierre el programa',
                          font=('Arial Black', 11))
        label3.place(relx=0.30, rely=0.55)

        tipo_erg = tk.StringVar()
        tipo_erg1 = AutocompleteCombobox(ventana1, textvar=tipo_erg, state='readonly',
                                       font=("Arial", 14), width=40)
        tipo_erg1["completevalues"] = ['RETIMAX (CSO)', 'RETeval (LKC Technologies)']
        tipo_erg1.place(relx=0.2, rely=0.45)
        def ventana3():
            ventana2 = tk.Frame(ventana1)
            ventana2.pack(expand=True, fill='both')
            fondo1 = tk.PhotoImage(file="img/retimax.gif")
            fondo1_1 = tk.Label(ventana2, image=fondo1).place(relx=0.15, rely=0.05)

            fondo2 = tk.PhotoImage(file="img/reteval.gif")
            fondo1_2 = tk.Label(ventana2, image=fondo2).place(relx=0.15, rely=0.4)

            label2 = tk.Label(ventana2, text='RETIMAX (CSO)', font=('Arial Black', 17))
            label2.place(relx=0.57, rely=0.2)

            label3 = tk.Label(ventana2, text='RETeval (LKC Technologies)', font=('Arial Black', 15))
            label3.place(relx=0.57, rely=0.6)

            boton1 = tk.Button(ventana2, cursor='hand2', text='Regresar', bg='light gray', font=('Arial black', 12),
                               command=ventana2.destroy)
            boton1.place(relx=0.02, rely=0.9)

            ventana2.mainloop()
        def ventana4():
            def getext():
                nombre_paciente = nombre.get()
                aparato_erg = tipo_erg.get()
                return [nombre_paciente, aparato_erg]
            ventana2 = tk.Frame(ventana1)
            ventana2.pack(expand=True, fill='both')
            if (getext()[0] == '' or getext()[1] ==''):
                w = 'No se seleccionaron los datos necesarios.\n' \
                    'Regrese y vuelva a intentarlo'
                label3 = tk.Label(ventana2, text=w, font=('Arial Black', 14))
                label3.place(relx=0.2, rely=0.3)
            else:
                try:
                    w = predicciones(getext()[0],getext()[1])
                    if w[0] == 0:
                        w1 = 'El ojo derecho tiene \nEdema Macular Diabético'
                    elif w[0] == 1:
                        w1 = 'El ojo derecho NO tiene \nEdema Macular Diabético'
                    else:
                        w1 = w[0]
                    if w[1] == 0:
                        w2 = 'El ojo izquierdo tiene \nEdema Macular Diabético'
                    elif w[1] == 1:
                        w2 = 'El ojo izquierdo NO tiene \nEdema Macular Diabético'
                    else:
                        w2 = w[1]

                    label3 = tk.Label(ventana2, text='Los resultados del paciente\n'+getext()[0]+' son:',
                                      font=('Arial Black', 15))
                    label3.place(relx=0.1, rely=0.1)

                    label4 = tk.Label(ventana2, text='OJO DERECHO',
                                      font=('Arial Black', 13))
                    label4.place(relx=0.1, rely=0.25)

                    label5 = tk.Label(ventana2, text='OJO IZQUIERDO',
                                      font=('Arial Black', 13))
                    label5.place(relx=0.6, rely=0.25)

                    label6 = tk.Label(ventana2, text=w1,
                                      font=('Arial', 13))
                    label6.place(relx=0.1, rely=0.35)

                    label7 = tk.Label(ventana2, text=w2,
                                      font=('Arial', 13))
                    label7.place(relx=0.6, rely=0.35)
                except:
                    w = '''Verifique que los datos estén en el formato solicitado
                    y que el aparato de ERG sea el correcto.
                    
                    Regrese y vuelva a intentarlo.'''
                    label3 = tk.Label(ventana2, text=w, font=('Arial Black', 14))
                    label3.place(relx=0.1, rely=0.3)


            boton1 = tk.Button(ventana2, cursor='hand2', text='Regresar', bg='light gray', font=('Arial black', 12),
                               command=ventana2.destroy)
            boton1.place(relx=0.02, rely=0.9)

            ventana2.mainloop()

        boton1 = tk.Button(ventana1, cursor='hand2', text='Regresar', bg='light gray', font=('Arial black', 12),
                            command=ventana1.destroy)
        boton1.place(relx=0.02, rely=0.9)

        boton3 = tk.Button(ventana1, cursor='hand2', text='¿?', bg='light gray', font=('Arial black', 12),
                           command=ventana3)
        boton3.place(relx=0.9, rely=0.35)

        boton2 = tk.Button(ventana1, cursor='hand2', text='VERIFICAR E\nINICIAR', bg='light gray', font=('Arial black', 16),
                            command=ventana4)
        boton2.place(relx=0.4, rely=0.7)


        ventana1.mainloop()

    boton11 = tk.Button(ventana_inicio, cursor='hand2', text='INICIAR', bg='light gray', font=('Arial black', 18),
                        command=ventana2)
    boton11.place(relx=0.4, rely=0.7)
    boton1 = tk.Button(ventana_inicio, cursor='hand2', text='Salir', bg='light gray', font=('Arial black', 12),
                       command=ventana_inicio.destroy)
    boton1.place(relx=0.02, rely=0.9)


    ventana_inicio.mainloop()

inicio()

