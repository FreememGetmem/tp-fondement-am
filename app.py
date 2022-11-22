import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(persist=True)
def load_data():
    data = datasets.load_iris()
    return data

def split(df, size, seed):
    y = df["target"]
    X = df.drop("target", axis =1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=round((size/100),2) ,random_state=seed)
    return X_train, X_test, y_train, y_test

def distribution(datas, xs,ys, markers):
    plt.scatter(data=datas, x=xs, y=ys, marker=markers)
    plt.xlabel(xs)
    plt.ylabel(ys)
    st.pyplot()

def modele_svm(modele, regul, ran_state, X, y, X_t, features):
    model = modele.LinearSVC(C=regul, random_state=ran_state)
    model.fit(X[features], y)
    y_pred = model.predict(X_t[features])
    val = classification_report(y,y_pred)
    return val, y_pred

def modele_svc(modele, regul, ran_state, X, y, X_t, features):
    model = modele.SVC(C=regul, random_state=ran_state)
    model.fit(X[features], y)
    y_pred = model.predict(X_t[features])
    val = classification_report(y,y_pred)
    return val, y_pred

def modele_tree(modele, depth, X, y, X_t, features):
    model = modele.SVC(C=regul, random_state=ran_state)
    model.fit(X[features], y)
    y_pred = model.predict(X_t[features])
    val = classification_report(y,y_pred)
    return val, y_pred

def confusion(y_t,y_p,df):
    mat_conf = confusion_matrix(y_t, y_p)
    sns.heatmap(mat_conf, square=True, annot=True, cbar=False
                , xticklabels=list(df.target_names)
                , yticklabels=list(df.target_names))
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    st.pyplot()

def confusion_normalise(y_t,y_p,df, norme):
    mat_conf = confusion_matrix(y_t, y_p, normalize=norme)
    sns.heatmap(mat_conf, square=True, annot=True, cbar=False
                , xticklabels=list(df.target_names)
                , yticklabels=list(df.target_names))
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    st.pyplot()

def fontiere_decision(X, Y, svms, reg, feature):
    # Prendre les deux premières features
    X = X[feature]
    y = Y

    # On crée une instance de SVM en fonction de leurs kernels
    models = ( svms.SVC(kernel="linear", C=reg), svms.LinearSVC(C=reg, max_iter=10000),
               svms.SVC(kernel="rbf", gamma=0.7, C=reg), svms.SVC(kernel="poly", degree=3, gamma="auto", C=reg), )
    models = (clf.fit(X, y) for clf in models)
    # Titre des des figures
    titles = ("SVC with linear kernel", "LinearSVC (linear kernel)", "SVC with RBF kernel", "SVC with polynomial (degree 3) kernel", )
    # On créer le grid 2x2
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0= X[feature[0]]
    X1 = X[feature[1]]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(clf,X,response_method="predict",cmap=plt.cm.coolwarm,alpha=0.8,
            ax=ax, xlabel=feature[0],ylabel=feature[1],)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    st.pyplot()

def modele_tree(m, depth,X, y, X_t, y_t,acc, pre, rec, dep):
    model =  m.DecisionTreeClassifier(max_depth = depth)
    model.fit(X, y)
    y_pred = model.predict(X_t)
    acc.append(model.score(X_t, y_t).round(3)*100)
    pre.append(precision_score(y_t, y_pred,pos_label='positive',average='micro').round(3)*100)
    rec.append(recall_score(y_t, y_pred,pos_label='positive',average='micro').round(3)*100)
    dep.append(round(depth,1))
    return acc, pre, rec, dep, y_pred

def arbre_decision(m,profond, echant, X, y,X_t, y_t):
    acc = []
    pre = []
    rec = []
    dep = []
    if len(profond) !=0:
        for max_depths in profond:
            accuracy,precision,recall,deep, y_pred =  modele_tree(m, max_depths,X, y, X_t, y_t,acc,pre,rec,dep)

        st.markdown("""### Taux de Classification par profondeur: """)
        data_acc = pd.DataFrame(accuracy, columns=['Accuracy'])
        data_pre = pd.DataFrame(precision, columns=['Précision'])
        data_rec = pd.DataFrame(recall, columns=['Recall'])
        data_deep = pd.DataFrame(deep, columns=['Profondeur'])
        data = pd.concat([data_deep,data_acc,data_pre,data_rec],axis=1).transpose()
        st.write(data)
        return y_pred

def main():
    st.title("TP d'application: Les fondements de l'AM")
    st.subheader("Auteur :  Mor NDOUR")
    seed = 123
    accuracy = []
    precision = []
    recall = []
    dep = []
    ech = []
    df = load_data()
    data = df.copy()
    data_all = df.copy()

    dfs = pd.DataFrame(df.data, columns=df['feature_names'])
    target =df.target
    dfs['target'] = target

    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu de données Iris : 150 observations")
        st.write(dfs)

    menu_deroulante = ["Etude des distributions", "Linear SVC", "SVC","Arbre de décision", "GridSearchCV"]
    page = st.sidebar.selectbox("Page de navigation", menu_deroulante)
    st.sidebar.markdown("""---""")

    if page == menu_deroulante[0]:
        st.markdown("Vous êtes sur la page Etude des distributions")
        size = st.sidebar.number_input("Choisir le pourcentage du test : ", 5, 100, step=10)
        features = st.sidebar.multiselect('Filtre des caractéristiques : ',df['feature_names'])

        cols1, cols2 = st.columns(2)

        if st.sidebar.button("Distribution", key='distribution'):
            X_train, X_test, y_train, y_test = split(df=dfs, size=size, seed=seed)
            with cols1:
                cols1.subheader("Données d'entrainement")
                distribution(X_train,features[0],features[1],'o')
            with cols2:
                cols2.subheader("Données de test")
                distribution(X_test,features[0],features[1],'+')

    elif page == menu_deroulante[1]:
        st.markdown("Vous êtes sur la page Linear SVC")
        size = st.sidebar.number_input("Choisir le pourcentage du test : ", 5, 100, step=10)
        features = st.sidebar.multiselect('Filtre des caractéristiques : ',df['feature_names'])
        regularisation = st.sidebar.number_input("Régularisation : ", 1, 10, step=1)
        random_state = st.sidebar.number_input("Random State : ", 1, 100, step=10)

        if st.sidebar.button("Exécution", key='execution'):
            X_train, X_test, y_train, y_test = split(df=dfs, size=size, seed=seed)
            m = svm
            val, y_pred = modele_svm(m, regularisation, random_state, X_train, y_train, X_test, features)

            st.markdown("""---""")
            st.markdown("""### Taux de Classification détaillé :""")
            st.text('\n ' + val)
            st.markdown("""---""")
            cols1, cols2 = st.columns(2)
            with cols1:
                st.markdown("""### Matrice de Confusion""")
                confusion(y_test,y_pred,df)
            with cols2:
                st.markdown("""### Matrice de Confusion normalisée""")
                confusion_normalise(y_test,y_pred,df, 'all')
            st.markdown("""---""")
            if len(features) <3:
                st.markdown("""### Graphe des frontières de Décision""")
                fontiere_decision(X_train, y_train, m, regularisation, features)

    elif page==menu_deroulante[2]:
        st.markdown("Vous êtes sur la page SVC")
        size = st.sidebar.number_input("Choisir le pourcentage du test : ", 5, 100, step=10)
        features = st.sidebar.multiselect('Filtre des caractéristiques : ',df['feature_names'])
        regularisation = st.sidebar.number_input("Régularisation : ", 1, 10, step=1)
        random_state = st.sidebar.number_input("Random State : ", 1, 100, step=10)

        if st.sidebar.button("Exécution", key='execution'):
            X_train, X_test, y_train, y_test = split(df=dfs, size=size, seed=seed)
            m = svm
            val, y_pred = modele_svc(m, regularisation, random_state, X_train, y_train, X_test, features)
            st.markdown("""---""")
            st.markdown("""### Taux de Classification détaillé :""")
            st.text('\n ' + val)
            cols1, cols2 = st.columns(2)
            st.markdown("""---""")
            with cols1:
                st.markdown("""### Matrice de Confusion""")
                confusion(y_test,y_pred,df)
            with cols2:
                st.markdown("""### Matrice de Confusion normalisée""")
                confusion_normalise(y_test,y_pred,df, 'all')

            st.markdown("""---""")
            st.markdown("""### Graphe des frontières de Décision""")
            fontiere_decision(X_train, y_train, m, regularisation, features)

    elif page==menu_deroulante[3]:
        st.markdown("Vous êtes sur la page Décision Tree")
        size = st.sidebar.number_input("Choisir le pourcentage du test : ", 5, 100, step=10)
        profondeur = st.sidebar.multiselect('Profondeur de l\'arbre : ', [1,2,3,4,5,6,7])
        echantillon = st.sidebar.multiselect('Echantillon par feuille : ', [2,3,5,10,15,20])

        if st.sidebar.button("Exécution", key='execution'):
            X_train, X_test, y_train, y_test = split(df=dfs, size=size, seed=seed)
            m = tree
            y_pred= arbre_decision(m,profondeur, echantillon, X_train, y_train,X_test, y_test)

            if len(profondeur) ==1:
                st.markdown("""### Matrice de Confusion""")
                mat_conf = confusion_matrix(y_test, y_pred)
                sns.heatmap(mat_conf, square=True, annot=True, cbar=False
                                    , xticklabels=list(df.target_names)
                                    , yticklabels=list(df.target_names))
                plt.xlabel('valeurs prédites')
                plt.ylabel('valeurs réelles')
                st.pyplot()

            if len(echantillon) !=0:
                for msplits in echantillon:
                    model =  tree.DecisionTreeClassifier(min_samples_split=msplits)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy.append(model.score(X_test, y_test).round(3)*100)
                    precision.append(precision_score(y_test, y_pred,pos_label='positive',average='micro').round(3)*100)
                    recall.append(recall_score(y_test, y_pred,pos_label='positive',average='micro').round(3)*100)
                    ech.append(round(msplits,1))

                st.markdown("""### Taux de Classification par echantillon: """)
                data_acc = pd.DataFrame(accuracy, columns=['Accuracy'])
                data_pre = pd.DataFrame(precision, columns=['Précision'])
                data_rec = pd.DataFrame(recall, columns=['Recall'])
                data_ech = pd.DataFrame(ech, columns=['Echantillon'])
                data = pd.concat([data_ech,data_acc,data_pre,data_rec],axis=1).transpose()
                st.write(data)

                if len(echantillon) ==1:
                    st.markdown("""### Matrice de Confusion""")
                    mat_conf = confusion_matrix(y_test, y_pred)
                    sns.heatmap(mat_conf, square=True, annot=True, cbar=False
                                    , xticklabels=list(df.target_names)
                                    , yticklabels=list(df.target_names))
                    plt.xlabel('valeurs prédites')
                    plt.ylabel('valeurs réelles')
                    st.pyplot()
                # st.write(profondeur)


    else:
        st.markdown("Vous êtes sur la page GridSearchCV")
        size = st.sidebar.number_input("Choisir le pourcentage du test : ", 5, 100, step=10)
        profondeur = st.sidebar.multiselect('Profondeur de l\'arbre : ', [1,2,3,4,5,6,7])
        echantillon = st.sidebar.multiselect('Echantillon par feuille : ', [2,3,5,10,15,20])
        if st.sidebar.button("Exécution", key='execution'):
            X_train, X_test, y_train, y_test = split(df=dfs, size=size, seed=seed)
            pgrid = {"max_depth": profondeur, "min_samples_split": echantillon}
            grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=pgrid, cv=2)
            grid_search.fit(X_train, y_train)


            # st.write(grid_search.best_estimator_.score(X_test, y_test))
            st.success("Valeur optimal pour la profondeur de l\'arbre : "+str(grid_search.best_params_["max_depth"]), icon="✅")
            st.success("Valeur optimal pour l\'échantillon par feuille  : "+str(grid_search.best_params_["min_samples_split"]), icon="✅")


if __name__ == '__main__':
    main()
