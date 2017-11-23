using Perceptrons

# learning a single target
X_train        = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
Y_train        = [1; 1; 1; 0.0]
X_test         = [.8 .9; .01 1; .9 0.2; 0.1 0.2]

model          = Perceptrons.fit(X_train,Y_train)
Y_pred         = Perceptrons.predict(model,X_test)

print("[Perceptron] accuracy : $(acc(Y_train,Y_pred))")

# if you want to save your model
Perceptrons.save(model,filename="/tmp/perceptron_model.jld")

# if you want to load back your model
model = Perceptrons.load(filename="/tmp/perceptron_model.jld")