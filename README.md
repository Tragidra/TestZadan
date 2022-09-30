# TestZadan
Я выполнил данное тестовое задание после его получения. Какие мысли у меня появились в ходе его выполнения? Ну, во-первых, выполнить его базовую составляющую, три пункта на считывание данных и их последующую обработку было несложно. Главная проблема заключалась в его последующей обработке – необходимо было придумать, как можно оптимизировать сам код и время выполнения программы, т.е. уменьшить количество ресурсов, затрачиваемых на выполнение поставленной задачи, уменьшить время, необходимое для получения результата. Первым делом я установил таймеры и засёк время работы программы, оно составляло в среднем около 6,3 секунд. После этого я начал копаться в коде решителей, посмотрел, как именно работает каждый солвер и что именно делает каждая строчка его кода. Я не решился вносить существенные изменения в структуру кода, поэтому оптимизировал несколько методов, поменял несколько рукописных методов на аналоги из библиотеки scipy, немного переделал ход работы программы. Это позволило уменьшить время выполнения кода до +-4,3 секунд. Следующим шагом был рефакторинг кода: я решил объединить все методы в один класс, дабы было удобнее с ним взаимодействовать, а всю загрузку и обработку данных расположил в методе-инициализации класса. Также я переделал третий солвер. так как при использовании его в едином классе с другими солверами отпадала нужда в повторной активации первых двух солверов (третий солвер в своей работе опирался на данные от первых двух солверов). Также я немного переделал структуру кода. Можно было ещё сильнее оптимизировать программу, например, переделать структуру солверов или заменить импорт всех библиотек на импорт одних только нужных в данной программе методов (это ускорило бы работу кода), однако, я посчитал это нецелесообразным, так как изменение самих солверов вряд ли подходило условиям задачи, а изменение импорта библиотек на импорт методов усложнило бы чтение кода. 
