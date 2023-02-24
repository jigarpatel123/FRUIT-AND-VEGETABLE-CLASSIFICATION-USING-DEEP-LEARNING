import tkinter
from tkinter import filedialog as fd
from PIL import ImageTk, Image 
from tkinter import ttk 
import cv2

master = tkinter.Tk() 
master.configure(bg='#B1ACA7')
master.title("Fruit and Vegetable Classification")
var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "blue",bg = "yellow",font = "Verdana 17 bold")
var.set("Fruit and Vegetable Classification")
label.pack()
def Train():
    argument=op.get()
    if argument=='Vegitable':
        import pathlib
        file = pathlib.Path("Model/Vegitable_360.h5")
        if file.exists ():
             tkinter.messagebox.showinfo(title="Done", message="Model Loaded")
        else:
            import keras
            from keras.preprocessing.image import ImageDataGenerator
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Flatten
            from keras.layers import Conv2D, MaxPooling2D
            batch_size = 32
            num_classes = 16
            epochs = 50
            model_name = "Model/Vegitable_360.h5"
            path_to_train = "Datasets/Vegitable/Training"
            path_to_test = "Datasets/Vegitable/Test"        
            Generator = ImageDataGenerator()
            train_data = Generator.flow_from_directory(path_to_train, (100, 100), batch_size=batch_size)        
            test_data = Generator.flow_from_directory(path_to_test, (100, 100), batch_size=batch_size)
            model = Sequential()
            model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(64, (5, 5),activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(128, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(num_classes, activation="softmax"))
            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
            model.fit_generator(train_data,
                                steps_per_epoch=1000//batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=test_data, validation_steps = 3)
            model.save(model_name)
    if argument=='Fruit':
        import pathlib
        file = pathlib.Path("Model/Vegitable_360.h5")
        if file.exists ():
             tkinter.messagebox.showinfo(title="Done", message="Model Loaded")
        else:
            import keras
            from keras.preprocessing.image import ImageDataGenerator
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Flatten
            from keras.layers import Conv2D, MaxPooling2D
            batch_size = 32
            num_classes = 21
            epochs = 50
            model_name = "Model/Fruit_360.h5"
            path_to_train = "Datasets/Fruit/Training"
            path_to_test = "Datasets/Fruit/Test"        
            Generator = ImageDataGenerator()
            train_data = Generator.flow_from_directory(path_to_train, (100, 100), batch_size=batch_size)        
            test_data = Generator.flow_from_directory(path_to_test, (100, 100), batch_size=batch_size)
            model = Sequential()
            model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(64, (5, 5),activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(128, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(num_classes, activation="softmax"))
            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
            model.fit_generator(train_data,
                                steps_per_epoch=1000//batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=test_data, validation_steps = 3)
            model.save(model_name)
            master.mainloop()  
def CaptureImage():
    import cv2
    camera = cv2.VideoCapture(0)
    while 1:
        return_value, image = camera.read()
        cv2.imshow("Frame",image)
        image=cv2.resize(image, (100, 100))
        cv2.imwrite("Model/query.jpg", image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    camera.release()
    del(camera)
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()
    img = Image.open("Model/query.jpg") 
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)    
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 5 + 1*30, width=250, height=250)
    
def BowseImage():
    name= fd.askopenfilename()
    img = Image.open(name) 
    img.save("Model/query.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)    
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 5 + 1*30, width=250, height=250)
    master.mainloop()    
def medfilt():
    img = cv2.imread("Model/query.jpg")
    img=cv2.medianBlur(img, 3)
    cv2.imwrite("Model/Pre.jpg",img)
    img = Image.open("Model/Pre.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)   
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 460, y = 5 + 1*30, width=250, height=250)    
    master.mainloop()   
def ApplyCNN():
    import cv2
    from keras.models import load_model
    img = cv2.imread("Model/query.jpg")
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img=img.reshape(1,100,100,3)
    argument=op.get()
    if argument=='Vegitable':
        model = load_model('Model/Vegitable_360.h5')
        model.summary()
        result=model.predict(img)
        label = {'Beetroot': 0,
                 'Cabsicum':1,
                 'Cauliflower': 2,
                 'Corn':3,
                 'Corn Husk':4,
                 'Eggplant':5,
                 'Fig':6,
                 'Ginger Root':7,
                 'Kohlrabi':8,
                 'Lemon':9,
                 'Limes':10,
                 'Onion Red': 11,
                 'Onion White':12,
                 'Potato White':13,
                 'Tomato Heart':14,
                 'Tomato Maroon':15}  
        label_d=list(label)
        result_classes = result.argmax(axis=-1)
        Class=label_d[result_classes[0]]
    if argument=='Fruit':
        model = load_model('Model/Fruit_360.h5')
        model.summary()
        label =  {'Apple Braeburn': 0,
                  'Banana': 1,
                  'Cherry 2':2,
                  'Cocos':3,
                  'Dates':4,
                  'Grape White':5,
                  'Guava':6,
                  'Hazelnut':7,
                  'Huckleberry':8,
                  'Kiwi':9,
                  'Lemon Meyer':10,
                  'Lychee':11,
                  'Mango Green':12,
                  'Mango Red':13,
                  'Orange':14,
                  'Papaya':15,
                  'Raspberry':16,
                  'Redcurrant':17,
                  'Strawberry':18,
                  'Walnut':19,
                  'Watermelon':20}  
        label_d=list(label)
        result=model.predict(img)
        result_classes = result.argmax(axis=-1)
        Class=label_d[result_classes[0]]
    var = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var, fg = "yellow",bg = "red",font = "Verdana 10 bold")
    var.set(Class)
    label.place(x = 10, y = 180 + 8*30, width=150, height=50)    
    master.mainloop()
 
     
def Exit():
    master.destroy()

master.geometry("750x500+100+100") 
master.resizable(width = True, height = True) 

op = ttk.Combobox(master,values=["Vegitable","Fruit"],font = "Verdana 10 bold")
op.place(x = 10, y = 5 + 1*30, width=150, height=50)
op.current(0)

b0 = tkinter.Button(master, text = "Train/Load Model", command = Train,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b0.place(x = 10, y = 30 + 2*30, width=150, height=50)

b1 = tkinter.Button(master, text = "Capture Image", command = CaptureImage,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b1.place(x = 10, y = 55 + 3*30, width=150, height=50)

b1 = tkinter.Button(master, text = "Upload Image", command = BowseImage,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b1.place(x = 10, y = 80 + 4*30, width=150, height=50)

b2 = tkinter.Button(master, text = "Pre-Process", command = medfilt,bg='#F1EAE3',fg='black',font = "Verdana 9 bold") 
b2.place(x = 10, y = 105 + 5*30, width=150, height=50)


b3 = tkinter.Button(master, text = "CNN Recognization", command = ApplyCNN,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b3.place(x = 10, y = 130 + 6*30, width=150, height=50)

b4 = tkinter.Button(master, text = "Quit", command = Exit,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b4.place(x = 10, y = 155 + 7*30, width=150, height=50)

var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "yellow",bg = "red",font = "Verdana 10 bold")
var.set("Class Name")
label.place(x = 10, y = 180 + 8*30, width=150, height=50)

master.mainloop() 
